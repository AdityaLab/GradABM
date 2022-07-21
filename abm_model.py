'''For training experiments: freeze values of all params except R0'''

''' Added with sparse logic '''

import os
import numpy as np
import pandas as pd
import torch
from data_utils import get_dir_from_path_list
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from abc import ABC, abstractmethod
from scipy.stats import gamma
import math
import torch.nn.functional as F
import yaml
INITIAL_INFECTED_RATIO = 0.5
INFINITY_TIME = np.inf # really large value outside the bounds of simulation steps 
USE_SPARSE = False

'''Utility Modules'''
class LogitRelaxedBernoulli(object):
    def __init__(self, logits, temperature=0.3, **kwargs):
        self.logits = logits
        self.temperature = temperature

    def rsample(self):
        eps = torch.clamp(
            torch.rand(
                self.logits.size(), dtype=self.logits.dtype, device=self.logits.device
            ),
            min=1e-6,
            max=1 - 1e-6,
        )
        y = (self.logits + torch.log(eps) - torch.log(1.0 - eps)) / self.temperature
        return y

    def log_prob(self, value):
        return (
            math.log(self.temperature)
            - self.temperature * value
            + self.logits
            - 2 * F.softplus(-self.temperature * value + self.logits)
        )

def lam(x_i, x_j, edge_attr, t, R, SFSusceptibility, SFInfector, lam_gamma_integrals):
        '''
            x_i and x_j are attributes from nodes for all edges in the graph
                note: x_j[:, 2].sum() will be higher than current infections because there are some nodes repeated
        '''
        S_A_s = SFSusceptibility[x_i[:,0].long()]
        A_s_i = SFInfector[x_j[:,1].long()]
        B_n = edge_attr[1, :]
        integrals = torch.zeros_like(B_n)
        infected_idx = x_j[:, 2].bool()
        infected_times = t - x_j[infected_idx, 3]
        
        integrals[infected_idx] =  lam_gamma_integrals[infected_times.long()]   #:,2 is infected index and :,3 is infected time
        edge_network_numbers = edge_attr[0, :] #to account for the fact that mean interactions start at 4th position of x
        I_bar = torch.gather(x_i[:, 4:27], 1, edge_network_numbers.view(-1,1).long()).view(-1) #to account for the fact that mean interactions start at 4th position of x
        res = R*S_A_s*A_s_i*B_n*integrals/I_bar #Edge attribute 1 is B_n

        return res.view(-1, 1)

class InfectionNetwork(MessagePassing):
    '''Contact network with graph message passing function for disease spread
    '''
    def __init__(self, lam, R, SFSusceptibility, SFInfector, lam_gamma_integrals, device):
        super(InfectionNetwork, self).__init__(aggr='add')
        self.lam = lam
        self.R = R 
        self.SFSusceptibility = SFSusceptibility 
        self.SFInfector = SFInfector 
        self.lam_gamma_integrals = lam_gamma_integrals
        self.device = device

    def forward_sparse(self, data, r0_value_trainable):
        x = data.x 
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        t = data.t
        # sparse adjacency matrix of inter-agent interactions
        S_A_s = self.SFSusceptibility[x[:,0].long()]
        A_s_i = self.SFInfector[x[:,1].long()]
        integrals = torch.zeros_like(S_A_s)
        infected_idx = x[:, 2].bool()
        infected_times = t - x[infected_idx, 3]
        integrals[infected_idx] =  self.lam_gamma_integrals[infected_times.long()]  #:,2 is infected index and :,3 is infected time
        I_bar = x[:, 4+22] # only info for random network being used in current expts
        integral_asi = A_s_i*integrals
        sparse_adj = torch.sparse_coo_tensor([edge_index[0,:].tolist(), edge_index[1,:].tolist()], torch.ones(edge_index.shape[1]).tolist(), (x.shape[0], x.shape[0])).to(self.device)
        sparse_asi = integral_asi.view(-1,1).to_sparse().to(self.device)
        sparse_mult = torch.sparse.mm(sparse_adj, sparse_asi)
        dense_mult = sparse_mult.to_dense().view(-1)
        
        # total infection
        infection_transmission = (r0_value_trainable*S_A_s*dense_mult)/I_bar #/I_bar
        return infection_transmission.view(1, -1)
                
    def forward(self, data, r0_value_trainable):
        x = data.x 
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        t = data.t
        return self.propagate(edge_index, x=x, edge_attr=edge_attr, t=t, 
                R=r0_value_trainable, SFSusceptibility=self.SFSusceptibility, 
                SFInfector=self.SFInfector, lam_gamma_integrals=self.lam_gamma_integrals)

    def message(self, x_i, x_j, edge_attr, t, 
                        R, SFSusceptibility, SFInfector, lam_gamma_integrals):
        # x_j has shape [E, in_channels]
        tmp = self.lam(x_i, x_j, edge_attr, t, 
            R, SFSusceptibility, SFInfector, lam_gamma_integrals)  # tmp has shape [E, 2 * in_channels]
        return tmp

class DiseaseProgression(ABC):
    ''' abstract class '''
    def __init__(self):
        pass

    @abstractmethod
    def initialize_variables(self):
        ''' initialize tensor variables depending on disease '''
        pass

    @abstractmethod
    def update_next_stage_times(self):
        ''' update time '''
        pass

    @abstractmethod
    def update_current_stage(self):
        ''' update stage '''
        pass

class SEIRMProgression(DiseaseProgression):
    '''SEIRM for COVID-19 '''
    def __init__(self,params):
        super(DiseaseProgression, self).__init__()
        # encoding of stages
        self.SUSCEPTIBLE_VAR = 0
        self.EXPOSED_VAR = 1 # exposed state
        self.INFECTED_VAR = 2
        self.RECOVERED_VAR = 3
        self.MORTALITY_VAR = 4
        # default times (only for initialization, later they are learned)
        self.EXPOSED_TO_INFECTED_TIME = 3
        self.INFECTED_TO_RECOVERED_TIME = 5
        # inf time
        self.INFINITY_TIME = params['num_steps'] + 1
        self.num_agents = params['num_agents']

    def initialize_variables(self,agents_infected_time,agents_stages,agents_next_stage_times):
        ''' initialize tensor variables depending on disease '''
        # agents in I have been at least a few days infected as they have been previously in exposed
        # assumption that agents in E have been infected 1 day, and agents in I have been infected EXPOSED_TO_INFECTED_TIME days
        agents_infected_time[agents_stages==self.EXPOSED_VAR] = -1
        agents_infected_time[agents_stages==self.INFECTED_VAR] = -1*self.EXPOSED_TO_INFECTED_TIME
        agents_next_stage_times[agents_stages==self.EXPOSED_VAR] = self.EXPOSED_TO_INFECTED_TIME
        agents_next_stage_times[agents_stages==self.INFECTED_VAR] = self.INFECTED_TO_RECOVERED_TIME

        return agents_infected_time, agents_next_stage_times

    def update_initial_times(self,learnable_params,agents_stages,agents_next_stage_times):
        ''' this is for the abm constructor '''
        infected_to_recovered_time = learnable_params['infected_to_recovered_time']
        exposed_to_infected_time = learnable_params['exposed_to_infected_time']
        agents_next_stage_times[agents_stages==self.EXPOSED_VAR] = exposed_to_infected_time
        agents_next_stage_times[agents_stages==self.INFECTED_VAR] = infected_to_recovered_time
        return agents_next_stage_times

    def get_newly_exposed(self,current_stages,potentially_exposed_today):
        # we now get the ones that new to exposure
        newly_exposed_today = (current_stages==self.SUSCEPTIBLE_VAR)*potentially_exposed_today
        return newly_exposed_today

    def update_next_stage_times(self,learnable_params,newly_exposed_today,current_stages, agents_next_stage_times, t):
        ''' update time '''
        exposed_to_infected_time = learnable_params['exposed_to_infected_time'] 
        infected_to_recovered_time = learnable_params['infected_to_recovered_time'] 
        # for non-exposed
        # if S, R, M -> set to default value; if E/I -> update time if your transition time arrived in the current time
        new_transition_times = torch.clone(agents_next_stage_times) 
        curr_stages = torch.clone(current_stages).long()
        new_transition_times[(curr_stages==self.INFECTED_VAR)*(agents_next_stage_times == t)] = self.INFINITY_TIME 
        new_transition_times[(curr_stages==self.EXPOSED_VAR)*(agents_next_stage_times == t)] = t+infected_to_recovered_time 
        return newly_exposed_today*(t+1+exposed_to_infected_time) + (1 - newly_exposed_today)*new_transition_times

    def get_target_variables(self,params,learnable_params,newly_exposed_today,current_stages,agents_next_stage_times,t):
        ''' get recovered (not longer infectious) + targets '''
        mortality_rate = learnable_params['mortality_rate'] 
        new_death_recovered_today = (current_stages * (current_stages == self.INFECTED_VAR)*(agents_next_stage_times <= t)) / self.INFECTED_VAR # agents when stage changes
        # update for newly recovered agents {recovered now}
        recovered_dead_now = new_death_recovered_today # binary bit vector
        NEW_DEATHS_TODAY = mortality_rate*new_death_recovered_today.sum()
        NEW_INFECTIONS_TODAY = newly_exposed_today.sum()

        return recovered_dead_now, NEW_INFECTIONS_TODAY, NEW_DEATHS_TODAY

    def update_current_stage(self,newly_exposed_today,current_stages,agents_next_stage_times, t):
        ''' progress disease: move agents to different disease stage '''
        transition_to_infected = self.INFECTED_VAR*(agents_next_stage_times <= t) + self.EXPOSED_VAR*(agents_next_stage_times > t)
        transition_to_mortality_or_recovered = self.RECOVERED_VAR*(agents_next_stage_times <= t) + self.INFECTED_VAR*(agents_next_stage_times > t) # can be stochastic --> recovered or mortality

        # Stage progression for agents NOT newly exposed today'''
        # if S -> stay S; if E/I -> see if time to transition has arrived; if R/M -> stay R/M
        stage_progression = (current_stages == self.SUSCEPTIBLE_VAR)*self.SUSCEPTIBLE_VAR \
            + (current_stages == self.RECOVERED_VAR)*self.RECOVERED_VAR + (current_stages == self.MORTALITY_VAR)*self.MORTALITY_VAR \
            + (current_stages == self.EXPOSED_VAR)*transition_to_infected \
            + (current_stages == self.INFECTED_VAR)*transition_to_mortality_or_recovered
        
        # update curr stage - if exposed at current step t or not
        current_stages = newly_exposed_today*self.EXPOSED_VAR + stage_progression
        return current_stages

    def init_stages(self, learnable_params, device):
        '''initial_infections_percentage should be between 0.1 to 1''' 
        initial_infections_percentage = learnable_params['initial_infections_percentage']
        prob_infected = (initial_infections_percentage/100) * torch.ones((self.num_agents,1)).to(device)
        p = torch.hstack((prob_infected,1-prob_infected))
        cat_logits = torch.log(p+1e-9)
        agents_stages = F.gumbel_softmax(logits=cat_logits,tau=1,hard=True,dim=1)[:,0]
        return agents_stages

class SIRSProgression(DiseaseProgression):
    '''SIRS for influenza '''
    def __init__(self,params):
        super(DiseaseProgression, self).__init__()
        # encoding of stages
        self.SUSCEPTIBLE_VAR = 0
        self.INFECTED_VAR = 1
        self.RECOVERED_VAR = 2
        # default times (only for initialization, later they are learned)
        self.INFECTED_TO_RECOVERED_TIME = 5
        self.RECOVERED_TO_SUSCEPTIBLE_TIME = 100
        # inf time
        self.INFINITY_TIME = params['num_steps'] + 1
        self.num_agents = params['num_agents']

    def initialize_variables(self,agents_infected_time,agents_stages,agents_next_stage_times):
        ''' initialize tensor variables depending on disease '''
        agents_infected_time[agents_stages==self.INFECTED_VAR] = -1
        agents_next_stage_times[agents_stages==self.INFECTED_VAR] = self.INFECTED_TO_RECOVERED_TIME
        return agents_infected_time, agents_next_stage_times
    
    def update_initial_times(self,learnable_params,agents_stages,agents_next_stage_times):
        infected_to_recovered_time = learnable_params['infected_to_recovered_time']
        ''' this is for the abm constructor '''
        agents_next_stage_times[agents_stages==self.INFECTED_VAR] = infected_to_recovered_time
        return agents_next_stage_times

    def get_newly_exposed(self,current_stages,potentially_exposed_today):
        # we now get the ones that new to exposure
        newly_exposed_today = (current_stages==self.SUSCEPTIBLE_VAR)*potentially_exposed_today
        return newly_exposed_today

    def update_next_stage_times(self,learnable_params,newly_exposed_today,current_stages, agents_next_stage_times, t):
        ''' update time '''
        infected_to_recovered_time = learnable_params['infected_to_recovered_time']
        recovered_to_susceptible_time = learnable_params['recovered_to_susceptible_time']
        # for non-exposed
        new_transition_times = torch.clone(agents_next_stage_times) 
        curr_stages = torch.clone(current_stages).long()
        new_transition_times[(curr_stages==self.INFECTED_VAR)*(agents_next_stage_times == t)] = t+recovered_to_susceptible_time
        new_transition_times[(curr_stages==self.RECOVERED_VAR)*(agents_next_stage_times == t)] = self.INFINITY_TIME  # they go back to susceptible

        return newly_exposed_today*(t+1+infected_to_recovered_time) + (1 - newly_exposed_today)*new_transition_times

    def get_target_variables(self,params,learnable_params,newly_exposed_today,current_stages,agents_next_stage_times,t):
        ''' get recovered (not longer infectious) + targets '''
        new_recovered_today = (current_stages * (current_stages == self.INFECTED_VAR)*(agents_next_stage_times <= t)) / self.INFECTED_VAR # agents when stage changes
        # get ILI
        # ILI =  dSI / self.N * 100 # multiply 100 because it is percentage
        # this is what Shaman and Pei do https://github.com/SenPei-CU/Multi-Pathogen_ILI_Forecast/blob/master/code/SIRS_AH.m
        ILI = newly_exposed_today.sum() / params['num_agents'] * 100
        NEW_INFECTIONS_TODAY = newly_exposed_today.sum()

        return new_recovered_today, NEW_INFECTIONS_TODAY, ILI

    def update_current_stage(self,newly_exposed_today,current_stages,agents_next_stage_times, t):
        ''' progress disease: move agents to different disease stage '''

        transition_to_recovered = self.RECOVERED_VAR*(agents_next_stage_times <= t) + self.INFECTED_VAR*(agents_next_stage_times > t) # can be stochastic --> recovered or mortality
        transition_to_susceptible = self.SUSCEPTIBLE_VAR*(agents_next_stage_times <= t) + self.RECOVERED_VAR*(agents_next_stage_times > t)

        # Stage progression for agents NOT newly exposed today'''
        # if S -> stay S; if E/I -> see if time to transition has arrived; if R/M -> stay R/M
        stage_progression = (current_stages == self.SUSCEPTIBLE_VAR)*self.SUSCEPTIBLE_VAR \
            + (current_stages == self.INFECTED_VAR)*transition_to_recovered \
            + (current_stages == self.RECOVERED_VAR)*transition_to_susceptible 
        
        current_stages = newly_exposed_today*self.INFECTED_VAR + stage_progression
        
        return current_stages

    def init_stages(self, learnable_params, device):
        '''initial_infections_percentage should be between 0.1 to 1''' 

        initial_infections_percentage = learnable_params['initial_infections_percentage']
        prob_infected = (initial_infections_percentage/100) * torch.ones((self.num_agents,1)).to(device)
        p = torch.hstack((prob_infected,1-prob_infected))
        cat_logits = torch.log(p+1e-9)
        agents_stages = F.gumbel_softmax(logits=cat_logits,tau=1,hard=True,dim=1)[:,0] 
        
        return agents_stages
    
class GradABM():
    def __init__(self, params, device):
        
        county_id = params['county_id']
        abm_params = f'Data/ABM_parameters/{county_id}_generated_params.yaml'

        #Reading params
        with open(abm_params, 'r') as stream:
            try:
                abm_params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print('Error in reading parameters file')
                print(exc)

        params.update(abm_params)
        self.params = params
        self.device = device
        self.num_agents = self.params['num_agents']
        print("Num Agents: ", self.num_agents)
                
        #**********************************************************************************
        #Environment state variables
        #**********************************************************************************
        #**********************************************************************************
        #Static
        self.agents_ix = torch.arange(0, self.params['num_agents']).long().to(self.device)
        infile = os.path.join(
                self.params['output_location']['parent_dir'],
                self.params['output_location']['agents_dir'],
                self.params['output_location']['agents_outfile'])
        
        agents_df = pd.read_csv(infile)
        self.agents_ages = torch.tensor(agents_df['age_group'].to_numpy()).long().to(self.device) #age constant in simulation 
        
        self.num_networks = 23
        self.network_type_dict = {}
        self.network_type_dict['random'] = 22
        self.network_type_dict_inv = {}
        self.network_type_dict_inv[22] = 'random'

        # select disease progression model 
        if params['disease'] == 'COVID':
            self.DPM = SEIRMProgression(params)
        elif params['disease'] == 'Flu':
            self.DPM= SIRSProgression(params)

        #Age and Network and Occupation may need to be checked to populate this
        self.agents_mean_interactions = 0*torch.ones(self.params['num_agents'], self.num_networks).to(self.device) 
        
        mean_int_ran_df = pd.read_csv('Data/Initialization/RandomNetworkParameters.csv')
        mean_int_ran_mu = torch.tensor(mean_int_ran_df['mu'].values).float().to(self.device)

        child_agents = self.agents_ages <= self.params['CHILD_Upper_Index']
        adult_agents = torch.logical_and(self.agents_ages > self.params['CHILD_Upper_Index'], self.agents_ages <= self.params['ADULT_Upper_Index']).view(-1)
        elderly_agents = self.agents_ages > self.params['ADULT_Upper_Index']
        self.agents_mean_interactions[child_agents.bool(), 22] = mean_int_ran_mu[0]
        self.agents_mean_interactions[adult_agents.bool(), 22] = mean_int_ran_mu[1]
        self.agents_mean_interactions[elderly_agents.bool(), 22] = mean_int_ran_mu[2]
        self.agents_mean_interactions_split = list(torch.split(self.agents_mean_interactions, 1, dim=1))
        self.agents_mean_interactions_split = [a.view(-1) for a in self.agents_mean_interactions_split]
        
        self.R = 5.18 # learnable, but the default value
        self.R = torch.tensor(self.R).to(self.device)
        if self.params['disease']=='COVID':
            self.SFSusceptibility  = torch.tensor([ 0.35, 0.69, 1.03, 1.03, 1.03, 1.03, 1.27, 1.52, 1.52]).float().to(self.device)
            self.SFInfector = torch.tensor([0.0, 0.33, 0.72, 0.0, 0.0]).float().to(self.device)
            self.lam_gamma = {}
            self.lam_gamma['scale'] = 5.5 
            self.lam_gamma['rate'] = 2.14 
        elif self.params['disease']=='Flu':
            # from CDC Table 1, median value: https://www.cdc.gov/flu/about/keyfacts.htm
            self.SFSusceptibility  = torch.tensor([13.2,7.9,7.4,7.4,7.4,12.0,12.0,3.9,3.9]).float().to(self.device)
            self.SFSusceptibility = F.softmax(self.SFSusceptibility,0) * 25  # normalize/scale
            self.SFInfector = torch.tensor([0.0, 0.72, 0.0]).float().to(self.device)
            self.lam_gamma = {}
            self.lam_gamma['scale'] = 7.5  # mean is scale/rate, CDC says it's 3-4 days
            self.lam_gamma['rate'] = 2.14 
        
        self.B_n = {}
        self.B_n['household'] = 2
        self.B_n['occupation'] = 1
        self.B_n['random'] = 1 #0.25
        
        self.lam_gamma_integrals = self._get_lam_gamma_integrals(**self.lam_gamma, t = self.params['num_steps'] + 10)  # add 10 to make sure we cover all
        self.lam_gamma_integrals = self.lam_gamma_integrals.to(self.device)
        self.net = InfectionNetwork(lam, self.R, self.SFSusceptibility, 
                                    self.SFInfector, self.lam_gamma_integrals, self.device).to(self.device) 
      
        self.current_time = 0
        #**********************************************************************************
        self.all_edgelist, self.all_edgeattr = self.init_interaction_graph(t=0) # get one initial interaction graph

    
    def init_interaction_graph(self, t):
        '''this is Part-1 of Step'''
        
        infile = os.path.join(get_dir_from_path_list(
            [self.params['output_location']['parent_dir'],
                self.params['output_location']['networks_dir'],
                self.params['output_location']['random_networks_dir']]
                ), '{}.csv'.format(t))
        
        random_network_edgelist_forward = torch.tensor(pd.read_csv(infile, header=None).to_numpy()).t().long()
        random_network_edgelist_backward = torch.vstack((random_network_edgelist_forward[1,:],
                                                         random_network_edgelist_forward[0,:]))
        random_network_edgelist = torch.hstack((random_network_edgelist_forward, 
                                                random_network_edgelist_backward))
        random_network_edgeattr_type = torch.ones(random_network_edgelist.shape[1]).long()*self.network_type_dict['random']

        random_network_edgeattr_B_n = torch.ones(random_network_edgelist.shape[1]).float()*self.B_n['random']
        random_network_edgeattr = torch.vstack((random_network_edgeattr_type, random_network_edgeattr_B_n))

        all_edgelist = torch.hstack((random_network_edgelist,))
        all_edgeattr = torch.hstack((random_network_edgeattr,))

        all_edgelist = all_edgelist.to(self.device)
        all_edgeattr = all_edgeattr.to(self.device)

        return all_edgelist, all_edgeattr
    
    def get_interaction_graph(self, t):
        return self.all_edgelist, self.all_edgeattr
    
    def init_state_tensors(self, learnable_params):
        ''' Initializing message passing network (currently no trainable parameters here) '''
        #Dynamic
        #a.Testing
        #b.Quarantine
        #c.Infection and Disease
        self.current_stages = self.DPM.init_stages(learnable_params, self.device)
        self.agents_infected_index = (self.current_stages > 0).to(self.device) #Not susceptible
        self.agents_infected_time = ((self.params['num_steps']+1)*torch.ones_like(self.current_stages)).to(self.device) #Practically infinite as np.inf gives wrong data type
        
        self.agents_next_stages = -1*torch.ones_like(self.current_stages).to(self.device)
        self.agents_next_stage_times = (self.params['num_steps']+1)*torch.ones_like(self.current_stages).long().to(self.device) #Practically infinite as np.inf gives wrong data type
        
        # update values depending on the disease
        self.agents_infected_time, self.agents_next_stage_times = \
                self.DPM.initialize_variables(self.agents_infected_time,self.current_stages,self.agents_next_stage_times)

        self.agents_next_stage_times = self.agents_next_stage_times.float()
        self.agents_infected_time = self.agents_infected_time.float()
    
    
    def step(self, t, param_t):
        '''Send as input: r0_value [hidden state] -> trainable parameters  and t is the time-step of simulation.'''
        # construct dictionary with trainable parameters
        learnable_params = {}
        if self.params['disease']=='COVID':
            learnable_params['r0_value'] = param_t[0]
            learnable_params['mortality_rate'] = param_t[1]
            learnable_params['initial_infections_percentage'] = param_t[2]
            learnable_params['exposed_to_infected_time'] = 3
            learnable_params['infected_to_recovered_time'] = 5
        elif self.params['disease']=='Flu':
            learnable_params['r0_value'] = param_t[0]
            learnable_params['initial_infections_percentage'] = param_t[1]
            learnable_params['infected_to_recovered_time'] = 3.5
            learnable_params['recovered_to_susceptible_time'] = 2000
        ''' change params that were set in constructor '''
        if t==0:
            self.init_state_tensors(learnable_params)
            self.agents_next_stage_times = self.DPM.update_initial_times(learnable_params,self.current_stages,self.agents_next_stage_times)
        
        #t = self.current_time
        '''Steps: i) Get interaction graph, ii) Message Passing of Infection iii) State Evolution '''
        
        # ******************************************************************************** #
        # Part-1. Interaction Graph - Output: EdgeList, EdgeFeatures and NodeFeatures
        all_edgelist, all_edgeattr = self.get_interaction_graph(t) # the interaction graphs for GNN at time t
        all_nodeattr = torch.stack((
                self.agents_ages,  #0
                self.current_stages.detach(),  #1 
                self.agents_infected_index.to(self.device), #2
                self.agents_infected_time.to(self.device), #3
                *self.agents_mean_interactions_split, #4 to 26
                torch.arange(self.params['num_agents']).to(self.device), #Agent ids (27)
        )).t()
        agents_data = Data(all_nodeattr, edge_index=all_edgelist, edge_attr=all_edgeattr, t=t, 
                           agents_mean_interactions = self.agents_mean_interactions)
        
        # ******************************************************************************** #
        # Part-2. Message Passing - Transmission Dynamics + New Infections: {GNN + Variational Inference} 
        # agent steps: i) collects infection [GNN]; ii) get infected based on total infection collected [Variational Inference]
        # message passing: collecting infection from neighbors
        lam_t = self.net(agents_data, learnable_params['r0_value'])
        prob_not_infected = torch.exp(-lam_t)
        p = torch.hstack((1-prob_not_infected,prob_not_infected))
        cat_logits = torch.log(p+1e-9)
        potentially_exposed_today = F.gumbel_softmax(logits=cat_logits,tau=1,hard=True,dim=1)[:,0]  # first column is prob of infections
        newly_exposed_today = self.DPM.get_newly_exposed(self.current_stages,potentially_exposed_today)

        # ******************************************************************************** #
        # Part-3. State Evolution -> Progression Dynamics {Deterministic}
        # to do here: i) update curr_stage; ii) update next_transition_time
        # self.agents_infected_time[t].sum()
        # check 2 things: i) got infected_today -> go from S to E; ii) already infected -> update E to I; I to R or M.
        # stage_progression, new_death_recovered_today = self.deterministic_stage_transition(self.agents_stages[t,:],
        
        # before updating, get target variables like new deaths or ILI
        # also get recovered ones 
        recovered_dead_now, target1, target2 = self.DPM.get_target_variables(self.params,learnable_params,newly_exposed_today,self.current_stages,self.agents_next_stage_times, t)
        
        # get next stages without updating yet the current_stages
        next_stages = self.DPM.update_current_stage(newly_exposed_today,self.current_stages,self.agents_next_stage_times, t)
        
        # update times with current_stages
        self.agents_next_stage_times = self.DPM.update_next_stage_times(learnable_params,newly_exposed_today,self.current_stages, self.agents_next_stage_times, t)
        
        # safely update current_stages
        self.current_stages = next_stages 

        # update for newly exposed agents {exposed_today}
        self.agents_infected_index[newly_exposed_today.bool()] = True
        self.agents_infected_time[newly_exposed_today.bool()] = t
        # remove recovered from infected indexes
        self.agents_infected_index[recovered_dead_now.bool()] = False

        # reconcile and return values
        self.current_time += 1
        return target1, target2
        
    def _get_lam_gamma_integrals(self, scale, rate, t):
        b = rate * rate / scale
        a = scale / b  # / b
        res = [(gamma.cdf(t_i, a=a, loc=0, scale=b) - gamma.cdf(t_i-1, a=a, loc=0, scale=b)) for t_i in range(t)]
        return torch.tensor(res).float()


if __name__ == "__main__":
    pass
