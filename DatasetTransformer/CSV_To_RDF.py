import pandas as pd #for handling csv and csv contents
from rdflib import Graph, Literal, RDF, URIRef, Namespace #basic RDF handling
# from rdflib.namespace import FOAF , XSD #most common namespaces
# import urllib.parse #for parsing strings to URI's
if __name__ == '__main__':
    data_df=pd.read_csv("/home/hussein/Downloads/DBLP_RandomEdgePairs - Sheet21.tsv",sep='\t')
    g = Graph()
    prefix = Namespace('https://dblp.org/')
    g.bind('dblp_prefix', prefix)
    S_URI=P_URI=O_URI=None
    for index, row in data_df.iterrows():
        if type(row[0]) is str and (row[0].startswith("http://") or row[0].startswith("https://")):
            S_URI=URIRef(row[0])
            if type(row[1]) is str and (row[1].startswith("http://") or row[1].startswith("https://")):
                P_URI = URIRef(row[1])
            else:
                P_URI = Literal(str(row[1]))

            if type(row[2]) is str and (row[2].startswith("http://") or row[2].startswith("https://")):
                O_URI = URIRef(row[2])
            else:
                O_URI = Literal(str(row[2]))
            g.add((S_URI, P_URI, O_URI))
        else:
            S_URI=Literal(str(row[0]))


    # print(g.serialize(format='turtle').decode('UTF-8'))
    g.serialize('DBLP_PrimaryAffaliationCountry.nt',format='n3')