import pandas as pd #for handling csv and csv contents
from rdflib import Graph, Literal, RDF, URIRef, Namespace #basic RDF handling
# from rdflib.namespace import FOAF , XSD #most common namespaces
# import urllib.parse #for parsing strings to URI's
import csv
if __name__ == '__main__':
    data_df=pd.read_csv("OGBL-CitiationV2.tsv",sep='\t')
    print("start s")
    data_df["s"]=data_df["s"].apply(lambda x : "<"+str(x)+">" if str(x).startswith("http") else "\""+str(x)+"\"")
    print("start p")
    data_df["p"] = data_df["p"].apply(lambda x: "<" + str(x) + ">" if str(x).startswith("http") else "\"" + str(x) + "\"")
    print("start o")
    data_df["o"] = data_df["o"].apply(lambda x: "<" + str(x) + ">" if str(x).startswith("http") else "\"" + str(x) + "\"")
    data_df["tripleEnd"]="."
    print("start writing the file")
    data_df.to_csv("OGBL-CitiationV2.nt",sep="\t",header=None,index=None,quoting=csv.QUOTE_NONE) ##prevent surronding text with quotes