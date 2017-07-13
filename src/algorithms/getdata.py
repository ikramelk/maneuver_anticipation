from elasticsearch import Elasticsearch
import redis



class GetData:
   

    # -------------------- Redis constants -------------------------------------- #
    REDIS_HOST = 'localhost'
    REDIS_PORT = 6379
    REDIS_DB = 0
    # ----------------- Setting up tools  --------------------------------------- #
    red = redis.StrictRedis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB
    )
        
    # ====================================================================== #
    pre_data = None
    def __init__(self):
        self.pre_data = self.data()

    

    def data(self):
        index = "clm_params"
        type = "dataset"
        size = 10000
        es = Elasticsearch([{
            'host': 'localhost',
            'port': 9200
        }])
        es.cluster.health(wait_for_status='yellow', request_timeout=1)
        if es.indices.exists(index=index):
            body = {
                
                "query": {
                    "match_all": {}
                }
            }
            res = es.search(index=index, doc_type=type, body=body)
            if res['hits']['total'] > 0:
                res_data = res['hits']['hits']
                data_source = []
                for ind in xrange(len(res_data)):
                    data_source.append(res_data[ind]['_source'])


        print('data_source')
        print(len(data_source))
        return data_source
