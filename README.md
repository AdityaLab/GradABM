# [AAMAS 2023] Differentiable Agent-based Epidemiology

## Publication

Implementation of the paper "Differentiable Agent-based Epidemiology."

Authors: Ayush Chopra*, Alexander Rodríguez*, Jayakumar Subramanian, Arnau Quera-Bofarull, Balaji Krishnamurthy, B. Aditya Prakash, Ramesh Raskar

*Equal contribution

Venue: AAMAS 2023 and preliminary version in AI4ABM Workshop @ ICML 2022 **(best paper award)**

Paper + appendix: [http://arxiv.org/abs/2207.09714](http://arxiv.org/abs/2207.09714)

## Requirements

Use the package manager [conda](https://docs.conda.io/en/latest/) to install required Python dependencies. Note: We used Python 3.7.

```bash
conda env create -f enviroment.yml
```

## Training

The following command will train and predict for all regions from epidemic week 202036 (GPU):

```bash
python -u main.py -st MA -j -d 0 1 2 3 -ew 202036 --seed 1234 -m GradABM-time-varying -di COVID
```

where `-st` is the US state (joint model for counties in the state ), `-j` is joint training, `-d` are the GPU devices to be used, `-ew` is the [epidemic week](https://epiweeks.readthedocs.io/en/stable/) and `di` is disease (either COVID or Flu).
For running this in multiple weeks, see examples in ```Scripts/run.sh```.

For CPU, you want to use:
```bash
python -u main.py -st MA -j -d cpu -ew 202036 --seed 1234 -m GradABM-time-varying -di COVID
```


## Contact:

If you have any questions about the code, please contact Alexander Rodriguez at arodriguezc[at]gatech[dot]edu and Ayush Chopra ayushc[at]mit[dot]edu 

## Cite our work
If you find our work useful, please cite our work:
- Ayush Chopra, Alexander Rodríguez, Jayakumar Subramanian, Arnau Quera-Bofarull, Balaji Krishnamurthy, B. Aditya Prakash, Ramesh Raskar. 2023. Differentiable Agent-based Epidemiology. In International Conference on Autonomous Agents and Multiagent Systems.

```
@inproceedings{chopra2023differentiable,
  title={Differentiable Agent-based Epidemiology},
  author={Chopra, Ayush and Rodr'\iguez, Alexander and Subramanian, Jayakumar and Quera-Bofarull, Arnau and Krishnamurthy, Balaji and Prakash, B Aditya and Raskar, Ramesh},
  booktitle={International Conference on Autonomous Agents and Multiagent Systems},
  year={2023}
}
