from fastapi import FastAPI 
import uvicorn
import logging
from pydantic import BaseModel, field_validator
from .recommender import recommender

logging.basicConfig(level=logging.INFO)
log= logging.getLogger(__name__)



class ALS_Request(BaseModel):
    user_id : int 
    n_recs : int = 5 

    @field_validator("user_id")
    @classmethod
    def user_id_must_positive(cls,v):
        if v<0:
            raise ValueError('user_id can not be negative')
        return v 
    
    @field_validator('n_recs')
    @classmethod
    def n_recs_must_positive(cls,v):
        if v<0:
            raise ValueError('n_recs can not be negative')
        return v 

app=FastAPI()
reco_engine=recommender()

@app.post('/recommendation')
async def get_recommendation(ALS_req : ALS_Request):
    try:
        recommendation_list,scores_list = reco_engine.search_item(ALS_req.user_id, ALS_req.n_recs)
        return {
                'recommendations':recommendation_list, 
                'scores':scores_list }
    except Exception as e:
        log.exception('Recommendation failed')
        return {
                'recommendations':[], 
                'scores':[]
        }





@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__=='__main__':
    uvicorn.run(app,host='0.0.0.0',port=8000)


