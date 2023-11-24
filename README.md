# KG-TOSA: A Task-Oriented graph SAmpler for GNN
<figure>
  <img src="KGTOSA_BGP.png" width="400" />
  <figcaption>Fig.1: The TOSG’s generic graph pattern is based on two parameters: (i) the direction (outgoing and incoming) predicates, and (i) the number of hops.</figcaption>
</figure>

<p><h3>KG-TOSA is the main sampling techniques utilized by <a href="https://github.com/CoDS-GCS/KGNET">KGNet</a> system.</h3></h3></p>
## Installation
* Clone the `KGTOSA` repo 
* Create `KGTOSA` Conda environment (Python 3.8) and install pip requirements.
* Activate the `KGTOSA` environment
```commandline
conda activate KGTOSA
```



<b>Extract TOSG triples:</b>
1. Node Classification
```python
python -u TOSG_Extraction/TOSG_Extraction_NC.py --sparql_endpoint http://206.12.98.118:8890/sparql --graph_uri http://dblp.org --target_rel_uri https://dblp.org/rdf/schema#publishedIn --TOSG d1h1 --batch_size 1000000 --out_file DBLP-15M_PV --threads_count 32  
```
2. Link Prediction
```python
python -u TOSG_Extraction/TOSG_Extraction_LP.py --target_rel_uri=isConnectedTo --data_path=<path> --dataset=YAGO3-10 --TOSG=d1h1 --file_sep=tab
```
<b>Transform NC TOSG dataset into PYG dataset</b>
```python
python -u DatasetTransformer/TSV_TO_PYG_dataset.py --traget_node_type=Paper --target_rel=publishedIn --csv_path=<path> --dataset_name=DBLP-15M_PV_d1h1 --file_sep=tab --split_rel=publish_year 
```
<b>Download KGTOSA NC datasets</b>
<li>
<a href="http://206.12.94.177/CodsData/KGNET/KGBen/MAG/MAG42M_PV_FG.zip">MAG_42M_PV_FG</a>
</li><li>
<a href="http://206.12.94.177/CodsData/KGNET/KGBen/MAG/MAG42M_PV_d1h1.zip">MAG_42M_PV_d1h1</a>
</li><li>
<a href="http://206.12.94.177/CodsData/KGNET/KGBen/DBLP/DBLP15M_PV_FG.zip">DBLP-15M_PV_FG</a>
</li><li>
<a href="http://206.12.94.177/CodsData/KGNET/KGBen/DBLP/DBLP15M_PV_d1h1.zip">DBLP-15M_PV_d1h1</a>
</li>
<li>
<a href="http://206.12.94.177/CodsData/KGNET/KGBen/YAGO/YAGO_FM200.zip">YAGO4-30M_PC_FG</a>
</li>
<li>
<a href="http://206.12.94.177/CodsData/KGNET/KGBen/YAGO/YAGO_Star200.zip">YAGO4-30M_PC_d1h1</a>
</li>

<b>Download KGTOSA LP datasets</b>
<li>
<a href="http://206.12.94.177/CodsData/KGNET/KGBen/YAGO3-10/KGTOSA_YAGO3-10.zip">YAGO3-10_FG_d2h1</a>
</li>
<li>
<a href="http://206.12.94.177/CodsData/KGNET/KGBen/OGBL-WikiKG2-2015/WikiKG2_LP.zip">WikiKG2_FG_d2h1</a>
</li>
<li>
<a href="http://206.12.94.177/CodsData/KGNET/KGBen/DBLP/LP/DBLP2023-010305.zip">DBLP2023-010305_FG_d2h1</a>
</li>
</p>



<b>Reproduce KGTOSA Results:</b>
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
  journal={Proceedings of the VLDB Endowment},
  publisher={CORR}
}
```