from pymongo import MongoClient

class MongoDBService:
    def __init__(self, uri: str, db_name: str, vector_search_index: str=None):
        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.vector_search_index = vector_search_index

    def insert_document(self, collection_name: str, document: dict):
        collection = self.db[collection_name]
        result = collection.insert_one(document)
        return result.inserted_id

    def find_documents(self, collection_name: str, query: dict):
        collection = self.db[collection_name]
        documents = collection.find(query)
        return list(documents)

    def update_document(self, collection_name: str, query: dict, update: dict):
        collection = self.db[collection_name]
        result = collection.update_one(query, {'$set': update})
        return result.modified_count

    def delete_document(self, collection_name: str, query: dict):
        collection = self.db[collection_name]
        result = collection.delete_one(query)
        return result.deleted_count

    def insert_many(self, collection_name: str, documents: list):
        collection = self.db[collection_name]
        result = collection.insert_many(documents)
        return result.inserted_ids
    
    def delete_duplicates(self, collection_name: str, unique_field: str):
        collection = self.db[collection_name]
        pipeline = [
            {"$group": {
                "_id": f"${unique_field}",
                "ids": {"$addToSet": "$_id"},
                "count": {"$sum": 1}
            }},
            {"$match": {
                "count": {"$gt": 1}
            }}
        ]
        duplicates = list(collection.aggregate(pipeline))
        deleted_count = 0
        for doc in duplicates:
            ids_to_delete = doc['ids'][1:]
    
    def vector_search(self, collection_name: str, query_vector: list, num_candidates=100, limit=10):
        collection = self.db[collection_name]
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_search_index,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": num_candidates,
                    "limit": limit
                }
            }
        ]
        results = collection.aggregate(pipeline)
        return list(results)
    
    def vector_search_filter(self, collection_name: str, query_vector: list, num_candidates=100, limit=10, nct_id=None):
        collection = self.db[collection_name]
        if not nct_id:
            return self.vector_search(collection_name, query_vector, num_candidates, limit)
        if nct_id:
            result = collection.find_one({"nctId": nct_id})
            if result:
                return [result]
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self.vector_search_index,
                    "path": "embedding",
                    "queryVector": query_vector,
                    "numCandidates": num_candidates,
                    "limit": limit,
                    "filter": { "nctId": nct_id }
                }
            }
        ]
        results = collection.aggregate(pipeline)
        return list(results)
    
    def filter_results(self, results: list, top_k: int = 5):
        return results[:top_k]
    
    