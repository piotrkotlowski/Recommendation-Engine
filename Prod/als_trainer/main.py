from fastapi import FastAPI
import uvicorn
import asyncio 
from .als_engine import ALSRecommender
from contextlib import asynccontextmanager
import logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


ACTIVE_COLLECTION = "items_embeddings_A"

IS_HEALTHY=True





async def periodic_als(interval_minutes: int = 30):
    global ACTIVE_COLLECTION, IS_HEALTHY
    interval_seconds = interval_minutes * 60
    while True:
        inactive = "items_embeddings_B" if ACTIVE_COLLECTION == "items_embeddings_A" else "items_embeddings_A"
        
        new_model = ALSRecommender(collection_name=inactive)
        try:
            log.info("Retraining!")

            new_model.run_pipeline() 

            IS_HEALTHY=True

            app.state.als=new_model

            ACTIVE_COLLECTION = inactive

            log.info("Finished Retraining!")
            log.info('Current collection: %s',ACTIVE_COLLECTION)

        except ALSRecommender.DataLoadException as e:
            log.info('Retraing failed: %s',e)
            IS_HEALTHY=False
            
        
        for remaining in range(interval_seconds,0,-10):
            log.info('Next retraining in %s',remaining)
            await asyncio.sleep(min(10,remaining))

@asynccontextmanager
async def lifespan(app: FastAPI):
   asyncio.create_task(periodic_als(interval_minutes=30))
   yield

app=FastAPI(lifespan=lifespan) 


@app.get('/health')
async def health():
    return {'status':'healthy' if IS_HEALTHY else 'unhealthy'}

@app.get('/collection_name')
async def get_collection_name():
    return {'collection_name': ACTIVE_COLLECTION}



if __name__=="__main__":
    uvicorn.run(app,host="0.0.0.0",port=8000)