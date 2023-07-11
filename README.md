# KG-TOSA: A Task-Oriented graph SAmpler for GNN
<figure>
  <img src="KGTOSA_BGP.png" width="400" />
  <figcaption>Fig.1: The TOSG’s generic graph pattern is based on two parameters: (i) the direction (outgoing and incoming) predicates, and (i) the number of hops.</figcaption>
</figure>

## Installation
* Clone the `KGTOSA` repo 
* Create `KGTOSA` Conda environment (Python 3.8) and install pip requirements.
* Activate the `KGTOSA` environment
```commandline
conda activate KGTOSA
```

<b>Reproduce Results:</b>
1. Node Claassifcation
```python
# run RGCN  
python rgcn-KGTOSA.py --Dataset <DatasetPath>
# run GraphSaint  
python graph_saint_KGTOSA.py --Dataset <DatasetPath>
# run ShaDowSaint  
python graph_saint_Shadow_KGTOSA.py --Dataset <DatasetPath>
# run SeHGNN  
python SeHGNN/ogbn/main.py --Dataset <DatasetPath>
```

2. Link Prediction <br>
extract the dataset folder under the data folder under each method path
```python
# run RGCN  
python RGCN/main.py --Dataset <DatasetName>
# run GraphSaint  
python Morse/main.py --dataset <DatasetName>
```


<p> KGTOSA SPARQL Variations</p>
<ul> 
<li>d1h1</li>
<li>d1h2</li>
<li>d2h1</li>
<li>d2h2</li>
<li>FG (full graph)</li>
<li>HG (Handcrafted graph)</li>
</ul>
