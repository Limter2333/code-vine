from fastapi import APIRouter

from code_vine.api.deps import SettingsDep

router = APIRouter()


@router.get("/health")
async def health_check(settings: SettingsDep) -> dict[str, str]:
    return {"status": "ok", "app": settings.app_name}


@router.post("/show")
async def show() -> dict[str, str]:
    return {"": ""}
