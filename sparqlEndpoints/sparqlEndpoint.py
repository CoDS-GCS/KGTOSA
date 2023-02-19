import pandas as pd
import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
import datetime
class sparqlEndpoint:
    def __init__(self):
        self.endpointUrl="http://localhost:6190/sparql"
#             self.endpoint="http://192.168.79.140:8890/sparql"            

#    Returns SparqlQuery As Dataframe
    def executeSparqlQuery(self,Sparql_query):
        # start_t = datetime.datetime.now()
        sparql = SPARQLWrapper(self.endpointUrl)
        sparql.setQuery(Sparql_query)
        sparql.setReturnFormat(JSON)
        results = sparql.query().convert()
        # end_t = datetime.datetime.now()
        # print("executeSparqlQuery Time=", end_t - start_t)
        # print(results)
        # start_t = datetime.datetime.now()
        lst_values=[]
        lst_all_rows=[]
        df=pd.DataFrame()
        if len(results["results"]["bindings"])>0:
            lst_columns = results["results"]["bindings"][0].keys()
            # print(type(results["results"]["bindings"]))
            for result in results["results"]["bindings"]:
                # print(result)
                for col in lst_columns:
                    lst_values.append(result[col]["value"])
                lst_all_rows.append(lst_values)
                # zipped = zip(lst_columns, lst_values)
                # a_dictionary = dict(zipped)
                lst_values=[]
        #         print(a_dictionary)
        #     df=df.append(a_dictionary,ignore_index=True)
            df=pd.DataFrame(lst_all_rows, columns=lst_columns)
        # end_t = datetime.datetime.now()
        # print("Query to dataframe Time=", end_t - start_t)
        return df 