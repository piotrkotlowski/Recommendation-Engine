from config import settings
from langchain_ollama import OllamaLLM
from langchain_core.prompts.chat import ChatPromptTemplate
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from langchain.output_parsers import OutputFixingParser
from .pydantic_helper import productQuery,isHistoryConnected, conversationType,conversationTypeEnum, QueryContext
import json 
import chromadb
from statistics import multimode, StatisticsError
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from .db import Database
from typing import List
import logging
import random
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - %(message)s"
)

class chatRecommendationSystem:
    def __init__(self,dsn:str=settings.PG_DSN,model:str=settings.OLLAMA_MODEL, 
                 base_url:str = settings.OLLAMA_URL,
                 chroma_host:str=settings.CHROMA_HOST,chroma_port:str=settings.CHROMA_PORT):
    
        self.database=Database(dsn)

        self.llm  = OllamaLLM(
                model=model,
                base_url=base_url,
                verbose=True, 
                temperature=0
                )
        
        self.client  = chromadb.HttpClient(host=chroma_host, port=chroma_port)

        self.embedding_function=  SentenceTransformerEmbeddingFunction(
                model_name=settings.RAG_EMB_MODEL
            )
        
        self.collection = self.client.get_or_create_collection(
            name=settings.RAG_CHROMA_COLLECTION,
            embedding_function=self.embedding_function
        )

        self.cross_encoder = CrossEncoder(settings.RERANK_MODEL)

        try:
            logger.info(f"List of collections: {self.client.list_collections()}")
            logger.info(f"Collection Name: {self.collection.name}")
            count = self.collection.count()
            logger.info(f"Number of documents in collection: {count}")
            if count > 0:
                sample = self.collection.get(include=["metadatas", "documents"], limit=5)
                logger.info("Sample documents:")
                for i, doc in enumerate(sample['documents']):
                    logger.info(f"{i+1}. ID: {sample['ids'][i]}, Document: {doc}, Metadata: {sample['metadatas'][i]}")
            else:
                logger.info("Collection is empty.")
        except Exception as e:
            logger.info(f"Failed to inspect collection: {e}")
    
    def _resolve_query_context(self, user_text: str, history: str):
        """
        Classify user query mode: 'new', 'more_category', 'product_followup'.
        """
        parser = PydanticOutputParser(pydantic_object=QueryContext)

        fixing_parser = OutputFixingParser.from_llm(
            llm=self.llm,
            parser=parser
        )

        prompt = PromptTemplate(
            template="""
            You are a conversation router.

            User query: "{user_text}"
            Conversation history: "{history}"

            Decide the mode:
            - "new": a new category or product search (e.g., "show me laptops").
            - "more_category": user wants more products of the SAME category as last time 
              (e.g., "show me more", "give me other options").
            - "product_followup": user asks about details, reviews, comparisons of 
              already shown products (e.g., "tell me more about product 2").

            Respond as JSON in this format:
            {format_instructions}
            """,
            input_variables=["user_text", "history"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        try:
            chain = prompt | self.llm | fixing_parser
            result = chain.invoke({"user_text": user_text, "history": history})

            return result.model_dump()["mode"]
        except Exception as e:
            logger.error(f"Error in _resolve_query_context: {e}")
            return "new"
    
    def _get_conversationType(self,user_text: str): 
        """
        Getting conversation type : chitchat or recommendaiton
        """
        parser=PydanticOutputParser(pydantic_object=conversationType) 

        fixing_parser=OutputFixingParser(parser=parser, llm=self.llm,retry_chain=True ) 

        prompt=PromptTemplate( 
                        template="""You are a strict classifier.
                Classify the user query into exactly one category:
                - "small_talk": greetings (hi, hello, hey), casual chat, jokes, or off-topic conversation.
                - "recommendation": ONLY if the user EXPLICITLY asks about products, reviews, comparisons, prices, or requests suggestions.

                IMPORTANT RULES:
                1. If the message looks like a greeting ("hi", "hello", "hey", "good morning"), or casual chat ("what’s up?", "how are you?"), ALWAYS classify as "small_talk".
                2. Only classify as "recommendation" if the user CLEARLY mentions a product or product-related need. 
                - Examples: "show me laptops", "compare iPhones", "recommend headphones under $200".
                3. DO NOT guess or assume hidden intent. If the query is ambiguous, default to "small_talk".

                Examples:
                - "Hi!" → small_talk
                - "What’s up?" → small_talk
                - "Show me laptops" → recommendation
                - "Can you recommend a phone?" → recommendation

                User query: "{user_text}"

                Respond ONLY with valid JSON in this format:
                {format_instruction}"""
                , 
              input_variables=['user_text'], 
              partial_variables={"format_instruction" : parser.get_format_instructions()} )
        try:
            chain = prompt | self.llm | fixing_parser 

            result=chain.invoke({"user_text" : user_text}) 

            result=result.model_dump()['type']

            return result
        except Exception as e:
            logger.error(f"Error in _get_conversationType: {e}")
            return conversationTypeEnum.small_talk

    def return_response(self, user_id: int, user_text: str) -> str:
        """"
        Main function for generating recommendation
        """

        history_text = self._get_summarized_history(user_id, user_text)

        convType = self._get_conversationType(user_text)
        logger.info('History: %s ',history_text)
        logger.info('ConvType: %s ', convType)
        if convType == conversationTypeEnum.recommendation:
            mode = self._resolve_query_context(user_text, history_text or "No history")
            logger.info("Mode: %s",mode)

            if mode == "new":
                products = self._get_products(user_id,user_text)
                if len(products) == 0:
                    return 'I did not find any products fill your demands.'
                template = """
                    You are a helpful assistant that recommends products.
                    The user is starting a **new search**.
                    Here are some products: {products}.
                    Based on the user request "{text}", recommend 3 products that best match.
                    Provide a short explanation for each recommended product.
                """

            elif mode == "more_category":
                last_category = self.database.get_last_category(user_id)
                if last_category:
                    products = self._get_products_by_category(last_category)
                else:
                    return "I don’t have a category from your last search. Can you specify again?"
                template = """
                    You are a helpful assistant that continues a recommendation session.
                    The user asked for **more products in the same category as before**.
                    Here are additional products: {products}.
                    Select 3 different options to suggest, avoiding repetition.
                    Provide a short explanation for each recommended product.
                """

            elif mode == "product_followup":
                past_products = self.database.get_last_shown_products(user_id)
                if not past_products:
                    return "I don’t have any previous products to follow up on."
                products = self._reuse_products(past_products)
                template = """
                    You are a helpful assistant that provides **follow-up details**.
                    The user wants to know more about products already shown.
                    Here are the products: {products}.
                    Based on the query "{text}", expand with details.
                """

        
            docs = products["documents"]
            metas = products["metadatas"]
            ids = products["ids"]

            if ids and isinstance(ids[0], list):
                ids = [i[0] for i in ids]

            reviews = self.database.get_product_reviews(product_ids=ids, limit=3)
            category_list = []
            products_dict = {}

            for doc, meta, id_ in zip(docs, metas, ids):
                product_info = {
                    "document": doc,
                    "metadata": meta,
                }

                if "category" in meta:
                    category_list.append(meta["category"])

                if reviews and str(id_) in reviews:
                    product_info["reviews"] = reviews[str(id_)]

                products_dict[id_] = product_info

            products_json = json.dumps(products_dict, ensure_ascii=False)

            prompt = ChatPromptTemplate.from_template(template=template)
            chain = prompt | self.llm
            response = chain.invoke({"products": products_json, "text": user_text})

            try:
                final_categories = multimode(category_list)
                final_category = final_categories[0] if final_categories else None
            except StatisticsError:
                final_category = None

            ids = list(ids)

            self.database.add_chat_history(
                user_id=user_id,
                message_text=user_text,
                response_text=response,
                recommendation_category=final_category,
                recommended_products=ids
            )
            return response


        else:
            template = "You are a friendly assistant for casual chat. Reply to: {text}"
            prompt = ChatPromptTemplate.from_template(template=template)
            chain = prompt | self.llm
            response = chain.invoke({"text": user_text})
            self.database.add_chat_history(user_id=user_id,message_text=user_text,response_text=response, 
                                           recommendation_category= None , recommended_products=None)
            
            return response



    def _reuse_products(self, product_ids: List[str]):
        """
        Helper. Finding information about previously searched items
        """
        try:
            return self.collection.get(ids=product_ids, include=["documents", "metadatas"])
        except Exception as e: 
            return "There are not any products to reuse" 
    

    def _get_products_by_category(self, category: str, n_results=6, pool_size=20):
        """
        Getting new products with random sampling
        """

        try:
            results = self.collection.query(
                query_texts=[category],
                n_results=pool_size,
                where={"category": category},
                include=["documents", "metadatas", "ids"]
            )
            indices = random.sample(range(len(results["ids"])), min(n_results, len(results["ids"])))
            return {
                "documents": [results["documents"][i] for i in indices],
                "metadatas": [results["metadatas"][i] for i in indices],
                "ids": [results["ids"][i] for i in indices],
            }
        except Exception as e:
            return "No more new products"
        

    def _get_products(self, user_id:int ,user_text: str, n_results: int = 3):
        """"
        Main function for retrieving product info from vector_db
        """
        max_results=50
        top_selected_products=15

        parser = PydanticOutputParser(pydantic_object=productQuery)
        fixing_parser = OutputFixingParser.from_llm(llm=self.llm, parser=parser)

        prompt = PromptTemplate(
            template="""
            You are an information extraction system.
            Your task is to read the user's product description and return exactly ONE JSON object following the schema below.

            Schema:
            {format_instructions}

            Extraction Rules:
            - "product": main product type (e.g., "laptop", "running shoes").
            - "brand": brand if mentioned, else null.
            - "max_price": budget ceiling, else null.
            - "min_rating": minimum rating, else null.
            If not mentioned, set field to null.

            User product description: "{query}"
            """,
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )
        try:
            chain = prompt | self.llm | fixing_parser
            filters = chain.invoke({'query': user_text})
            filters_dict = filters.model_dump(exclude_none=True)

            query_text = filters_dict.pop("product", user_text)
        except Exception as e:
            logger.error(f"Error in _get_products: {e}")
            query_text = user_text
            filters_dict = {}

        where_clause = {}
        if "brand" in filters_dict:
            where_clause["brand"] = filters_dict["brand"]
        if "max_price" in filters_dict:
            where_clause["price"] = {"$lte": filters_dict["max_price"]}
        if "min_rating" in filters_dict:
            where_clause["rating"] = {"$gte": filters_dict["min_rating"]}

        results = self.collection.query(
            query_texts=[query_text],
            n_results=max_results,
            where=where_clause if where_clause else None
        )

        ids = []
        if results and "ids" in results and results["ids"]:
            ids = results["ids"][0] or []

        if ids==[]:
            return []
        else:
            purchased_products=self.database.get_purchased_items(user_id=user_id)
            
            final_ids=[]
            break_flag=False
            for pid in ids:
                if pid not in purchased_products:
                    final_ids.append(pid)
                if len(final_ids) >= top_selected_products:
                    break_flag=True
                    break
            if break_flag==False: 
                final_ids=ids[:top_selected_products]
            
            filtered_results = self.collection.get(
                ids=final_ids,
                include=["documents", "metadatas"])
            
            return self._get_rerank(user_text,filtered_results,n_results)
                                   

    def _get_summarized_history(self, user_id: int, user_text: str) -> str:
        """
        Returning summerized previous conversations.
        """

        history = self.database.get_users_history(user_id, limit=5)

        if not history:
            return None 

        history_text = "\n".join([f"User: {item['message_text']}\nAssistant: {item['response_text']}" for item in history])

        parser = PydanticOutputParser(pydantic_object=isHistoryConnected)

        prompt = PromptTemplate(
            template="""
            You are an AI assistant tasked with analyzing conversation context.
            Determine whether the previous conversation ("{history}") is related to the new user request ("{new_text}").

            Criteria for connection:
            - Continuation of topic, product, or user goal.
            - Not a completely unrelated request.

            Your answer must be a JSON object with one key "connection" and value true or false.
            {format_instructions}
            """, 
            input_variables=["new_text", "history"],
            partial_variables={"format_instructions": parser.get_format_instructions()}
        )
        try:
            chain = prompt | self.llm | parser 
            result = chain.invoke({'new_text': user_text, "history": history_text})
            result_dict = result.model_dump()
        except:
            return None 

        if not result_dict.get("connection", False):
            return None  
        
        template = """
        You are a helpful assistant that summarizes chat history between a user and an AI assistant.
        Here is the chat history: {history}.
        Summarize the chat history, focusing on the user's preferences and needs.
        """
        prompt = ChatPromptTemplate.from_template(template=template)
        chain = prompt | self.llm
        response = chain.invoke({"history": history_text})
        return response

    def _get_rerank(self, user_text: str, products: dict, n_results: int) -> dict:
        """
        Rerank products using cross-encoder, returning top-n results.
        """
        
        documents = products.get("documents", [])
        metadatas = products.get("metadatas", [])
        ids = products.get("ids", [])

        if not documents or not metadatas or not ids:
            return {"documents": [], "metadatas": [], "ids": []}
          
        scores = self.cross_encoder.predict([(user_text, doc) for doc in documents])

        logger.info(f"Rerank scores: {scores}")

        ranked_products = [
            prod_tuple for _, prod_tuple in sorted(
                zip(scores, zip(documents, metadatas, ids)), reverse=True
            )
        ][:n_results]

        if not ranked_products:
            return {"documents": [], "metadatas": [], "ids": []}

        documents, metadatas, ids = zip(*ranked_products)
        return {"documents": documents, "metadatas": metadatas, "ids": ids}

