from fastapi import APIRouter, HTTPException

async def health_check():
    return {"message": "success"}
