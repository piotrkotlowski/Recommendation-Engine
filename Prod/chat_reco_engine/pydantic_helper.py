from pydantic import BaseModel 
from typing import Literal
from enum import Enum


class conversationTypeEnum(str,Enum):
    review = "review"
    chichat= "chitchat"
    recommendation = "recommendation"

class conversationType(BaseModel):
     type : conversationTypeEnum

class QueryContext(BaseModel):
    mode: Literal['new','more_category','product_followup']

class productQuery(BaseModel):
    product: str | None = None 
    brand: str | None = None
    max_price: float | None = None
    min_rating: float | None = None

class isHistoryConnected(BaseModel):
    connection : bool