from typing import Annotated

from fastapi import Depends

from code_vine.core.config import Settings, get_settings

SettingsDep = Annotated[Settings, Depends(get_settings)]
