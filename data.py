import dataclasses as dc
import json
from typing import (
    Any,
    Collection,
    Mapping,
    Type,
    TypeVar,
)

from util.reflection import is_collection_constructible, parse_typehint, TypeInfo


__all__ = ["json_to_dataclass", "dict_to_dataclass"]


T = TypeVar("T")


def json_to_dataclass(cls: Type[T], s: str) -> T:
    if not dc.is_dataclass(cls):
        raise TypeError

    return dict_to_dataclass(cls, json.loads(s))


def dict_to_dataclass(cls: Type[T], raw: dict) -> T:
    if not dc.is_dataclass(cls):
        raise TypeError

    return _construct_dataclass(cls, raw)


def _construct_object(type_info: TypeInfo, raw_value: Any):
    if raw_value is None:
        if type_info.is_optional:
            return None
        raise TypeError

    if type_info.is_real_collection:
        if not is_collection_constructible(type_info.real_type, type(raw_value)):
            raise TypeError

        if not type_info.is_generic:
            return type_info.cast(raw_value)

        if type_info.is_mapping:
            return _construct_generic_mapping(type_info, raw_value)
        else:
            return _construct_generic_collection(type_info, raw_value)

    if type_info.is_dataclass:
        if not isinstance(raw_value, Mapping):
            raise TypeError
        return _construct_dataclass(type_info.real_type, raw_value)

    return type_info.cast(raw_value)


def _construct_generic_collection(type_info: TypeInfo, raw_value: Collection):
    eti = type_info.element_type_info
    return type_info.create(_construct_object(eti, elem) for elem in raw_value)


def _construct_generic_mapping(type_info: TypeInfo, raw_value: Mapping):
    kti, vti = type_info.key_type_info, type_info.value_type_info
    return type_info.create(
        (_construct_object(kti, key), _construct_object(vti, value))
        for key, value in raw_value.items()
    )


Missing = object()


def _construct_dataclass(cls: Type[T], raw_obj: Mapping) -> T:
    constructed = {}
    for field in dc.fields(cls):
        type_info = parse_typehint(field.type)
        raw_value = raw_obj.get(field.name, Missing)
        if raw_value is Missing:
            if field.default is not dc.MISSING:
                raw_value = field.default
            elif field.default_factory is not dc.MISSING:
                raw_value = field.default_factory()
            else:
                raise TypeError

        constructed[field.name] = _construct_object(type_info, raw_value)

    return cls(**constructed)
