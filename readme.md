<!--  <p align="center" style="font-size:50px;"> -->
# Official Implementation of GLoCE 
<!-- </p> -->

## Setup for experiments

**OS**: Ubuntu 20.04.4 LTS

**Python**: 3.9.19

<pre>
conda create -n CoMEL python=3.9
</pre>

Please install packages in requirements.txt
<pre>
pip install -r requirements.txt
</pre>


## Data Preparation

We followed the data preparation protocol by CLAM:
https://github.com/mahmoodlab/CLAM


Data for CAMELYON-16, PAIP, and TCGA are available:

CAMELYON-16:
https://camelyon16.grand-challenge.org/Data/

PAIP:
http://www.wisepaip.org/

TCGA:
https://portal.gdc.cancer.gov/


## Running Experiments
### Continual Instance classification

<pre>
cd ./continual
bash ./shell_scripts/continual_instance/run_cdatmil_ppl_owlora.sh
</pre>

- Other continual learning methods can be run: EWC, LwF, ER, DER++, InfLoRA, etc

### Continual Bag classification

<pre>
cd ./continual
bash ./shell_scripts/continual_bag/run_cdatmil_ppl_owlora.sh
</pre>

- Other continual learning methods can be run: EWC, LwF, ER, DER++, InfLoRA, etc

### Joint Instance classification

<pre>
cd ./joint
bash ./shell_scripts/joint_instance/run_cdatmil_ppl.sh
</pre>

- Other continual learning methods can be run: ABMIL, DSMIL, TransMIL, RRTMIL, etc
