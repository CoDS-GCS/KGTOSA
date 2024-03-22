## KGTOSA: Task-Oriented GNNs Training on Large Knowledge Graphs for Accurate and Efficient Modeling. 
<p style="color:blue">Accepted ICDE-2024</p>
<p>Hussein Abdallah, Walid Afandi, Panos Kalnis, and Essam Mansour
<br>
Contact: Hussein Abdallah (hussein.abdallah@mail.conocrdia.ca)
</p>
<a href="https://arxiv.org/abs/2403.05752">Latest version of the paper.</a>
<p>Abstract: A Knowledge Graph (KG) is a heterogeneous graph encompassing a diverse range of node and edge types. Heterogeneous Graph Neural Networks (HGNNs) are popular for training machine learning tasks like node classification and link prediction on KGs. However, HGNN methods exhibit excessive complexity influenced by the KG’s size, density, and the number of node and edge types. AI practitioners handcraft a subgraph of a KG G relevant to a specific task. We refer to this subgraph as a task-oriented subgraph (TOSG), which contains a subset of taskrelated node and edge types in G. Training the task using TOSG instead of G alleviates the excessive computation required for a large KG. Crafting the TOSG demands a deep understanding of the KG’s structure and the task’s objectives. Hence, it is challenging and time-consuming. This paper proposes KG-TOSA, an approach to automate the TOSG extraction for task-oriented HGNN training on a large KG. In KG-TOSA, we define a generic graph pattern that captures the KG’s local and global structure relevant to a specific task. We explore different techniques to extract subgraphs matching our graph pattern: namely (i) two techniques sampling around targeted nodes using biased random walk or influence scores, and (ii) a SPARQL-based extraction method leveraging RDF engines’ built-in indices. Hence, it achieves negligible preprocessing overhead compared to the sampling techniques. We develop a benchmark of real KGs of large sizes and various tasks for node classification and link prediction. Our experiments show that KG-TOSA helps state-of-the-art HGNN methods reduce training time and memory usage by up to 70% while improving the model performance, e.g., accuracy and inference time.</p>
<center>
<figure>
  <img src="KGTOSA_BGP.png" width="400" />
  <figcaption>Fig.1: The TOSG’s generic graph pattern is based on two parameters: (i) the direction (outgoing and incoming) predicates, and (i) the number of hops.</figcaption>
</figure>
</center>
<p><h3>KGTOSA is the HGNN sampler utilized by KGNet system (<a href="https://github.com/CoDS-GCS/KGNET">Published at ICDE2023</a>).</h3> </h3></p>

## Installation
* Clone the `KGTOSA` repo 
* Create `KGTOSA` Conda environment (Python 3.8) and install pip requirements.
* Activate the `KGTOSA` environment
```commandline
conda activate KGTOSA
```
## KGTOSA and Full-graph Datasets
These datasets are extracted from the knoweldge graph using SPARQL Queries and transformed into PYG dataloader format.
The d1h1 datasets are extrated using the KGTOSA Algo.3 (<a href="https://arxiv.org/abs/2403.05752">here</a>). 
### Download the ready datasets below
<b>Download KGTOSA NC datasets</b>
<li>
<a href="http://206.12.102.56/CodsData/KGNET/KGBen/MAG/MAG42M_PV_FG.zip">MAG_42M_PV_FG</a>
</li><li>
<a href="http://206.12.102.56/CodsData/KGNET/KGBen/MAG/MAG42M_PV_d1h1.zip">MAG_42M_PV_d1h1</a>
</li><li>
<a href="http://206.12.102.56/CodsData/KGNET/KGBen/DBLP/DBLP15M_PV_FG.zip">DBLP-15M_PV_FG</a>
</li><li>
<a href="http://206.12.102.56/CodsData/KGNET/KGBen/DBLP/DBLP15M_PV_d1h1.zip">DBLP-15M_PV_d1h1</a>
</li>
<li>
<a href="http://206.12.102.56/CodsData/KGNET/KGBen/YAGO/YAGO_FM200.zip">YAGO4-30M_PC_FG</a>
</li>
<li>
<a href="http://206.12.102.56/CodsData/KGNET/KGBen/YAGO/YAGO_Star200.zip">YAGO4-30M_PC_d1h1</a>
</li>

<b>Download KGTOSA LP datasets</b>
<li>
<a href="http://206.12.102.56/CodsData/KGNET/KGBen/YAGO3-10/KGTOSA_YAGO3-10.zip">YAGO3-10_FG_d2h1</a>
</li>
<li>
<a href="http://206.12.102.56/CodsData/KGNET/KGBen/OGBL-WikiKG2-2015/WikiKG2_LP.zip">WikiKG2_FG_d2h1</a>
</li>
<li>
<a href="http://206.12.102.56/CodsData/KGNET/KGBen/DBLP/LP/DBLP2023-010305.zip">DBLP2023-010305_FG_d2h1</a>
</li>

### OR

<b>Extract and Transform the dataset triples:</b>
1. Node Classification
```python
python -u TOSG_Extraction/TOSG_Extraction_NC.py --sparql_endpoint http://206.12.98.118:8890/sparql --graph_uri http://dblp.org --target_rel_uri https://dblp.org/rdf/schema#publishedIn --TOSG d1h1 --batch_size 1000000 --out_file DBLP-15M_PV --threads_count 32  
```
2. Link Prediction
```python
python -u TOSG_Extraction/TOSG_Extraction_LP.py --target_rel_uri=isConnectedTo --data_path=<path> --dataset=YAGO3-10 --TOSG=d1h1 --file_sep=tab
```
#### Transform NC TOSG dataset into PYG dataset
```python
python -u DatasetTransformer/TSV_TO_PYG_dataset.py --traget_node_type=Paper --target_rel=publishedIn --csv_path=<path> --dataset_name=DBLP-15M_PV_d1h1 --file_sep=tab --split_rel=publish_year 
```




## Train your Model:
1. Node Classification
```python
# run RGCN  
python rgcn-KGTOSA.py --Dataset <DatasetPath>
# run GraphSaint  
python graph_saint_KGTOSA.py --Dataset <DatasetPath>
# run ShaDowSaint  
python graph_saint_Shadow_KGTOSA.py --Dataset <DatasetPath>
# run SeHGNN  
python SeHGNN/ogbn/main.py --Dataset <DatasetPath>
# run IBS
python  IBS/run_ogbn_ppr.py --with config/<Config_path>  
```

2. Link Prediction <br>
extract the dataset folder under the data folder under each method path
```python
# run RGCN  
python RGCN/main.py --Dataset <DatasetName> --TargetRel <target_rel>
# run MorsE  
python Morse/main.py --dataset <DatasetName> --TargetRel <target_rel
# run LHGNN  
python LHGNN/main.py --dataset <DatasetName> --TargetRel <target_rel
```

## Citing Our Work
If you find our work useful, please cite it in your research:
<br>
```html
@article{KGTOSA,
  title={Task-Oriented GNNs Training on Large Knowledge Graphs for Accurate and Efficient Modeling},
  author={Abdallah, Hussein and Afandi, Waleed and Kalnis, Panos and Mansour, Essam},
  booktitle={2024 IEEE 40th International Conference on Data Engineering (ICDE)}, 
  year={2024},
}
```
