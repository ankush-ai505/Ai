from fastapi import FastAPI
from app.api.v1 import products, cart, orders, payments

app = FastAPI()

app.include_router(products.router, prefix='/api/v1/products', tags=['products'])
app.include_router(cart.router, prefix='/api/v1/cart', tags=['cart'])
app.include_router(orders.router, prefix='/api/v1/orders', tags=['orders'])
app.include_router(payments.router, prefix='/api/v1/payments', tags=['payments'])
