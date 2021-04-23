import dataclasses as dc
import datetime as dt
import enum
import json
from decimal import Decimal
from functools import partial
from types import FunctionType
from typing import Any, Callable, Collection, Dict, Type, Union
from uuid import UUID

import typing


__all__ = [
    "RecordSerializer",
    "load_json",
    "load_dict",
    "dump_json",
    "register_type",
    "register_dataclass",
]

from shared.json_util import (
    datetime_to_json,
    json_to_datetime,
    json_to_time,
    time_to_json,
)


class RecordSerializer:
    def __init__(self):
        self._extra_types: Dict[Type, Dict[str, Callable]] = {}
        self.register_type(UUID, str, UUID)
        self.register_type(Decimal, str, Decimal)
        self.register_type(dt.datetime, datetime_to_json, json_to_datetime)
        self.register_type(dt.time, time_to_json, json_to_time)
        self.register_type(
            dt.date, lambda d: d.isoformat(), lambda s: dt.date.fromisoformat(s)
        )
        self.register_type(set, list, set)
        self.register_type(frozenset, list, frozenset)

    def register_type(self, type_: Type, encoder: Callable, decoder: Callable):
        self._extra_types[type_] = {
            "encoder": encoder,
            "decoder": decoder,
        }

    def register_dataclass(self, class_type: Type):
        self._make_dataclass_decoder(class_type)

    def load_json(self, class_type: Type, raw: str, **kwargs) -> Any:
        return self.load_dict(class_type, json.loads(raw, **kwargs))

    def load_dict(self, class_type: Type, raw: dict) -> Any:
        decoder = self._make_dataclass_decoder(class_type)
        return decoder(raw)

    def dump_json(self, obj: Any, **kwargs) -> str:
        # Make sure the encoder is not accidentally overwritten
        kwargs.pop("default", None)
        return json.dumps(obj, default=self._encode_object, **kwargs)

    def _encode_object(self, obj: Any) -> Any:
        entry = self._extra_types.get(type(obj))
        if entry:
            return entry["encoder"](obj)
        elif isinstance(obj, enum.Enum):
            return obj.value
        elif dc.is_dataclass(obj):
            return dc.asdict(obj)

        raise TypeError(f"Unsupported type {type(obj)}")

    def _make_dataclass_decoder(self, class_type: Type):
        class_type = self._unwrap_alias(class_type)

        if not dc.is_dataclass(class_type):
            raise TypeError(f"{class_type} is not a dataclass")

        if class_type in self._extra_types:
            return self._extra_types[class_type]["decoder"]

        decoders = {}
        field: dc.Field
        # noinspection PyDataclass
        for field in dc.fields(class_type):
            field_type = self._unwrap_alias(field.type)
            original_type = self._unwrap_alias(typing.get_origin(field_type))
            if original_type is None:
                decoders[field.name] = self._make_scalar_decoder(field_type)
            elif original_type is Union:
                type_args = typing.get_args(field_type)
                if len(type_args) != 2 or type_args[1] is not type(None):
                    raise TypeError(
                        f"Unsupported type {field_type}, Union is not allowed unless"
                        f" it represents Optional[T]"
                    )
                inner_type = self._unwrap_alias(type_args[0])
                original_inner_type = typing.get_origin(inner_type)
                if original_inner_type is None:
                    decoders[field.name] = self._make_scalar_decoder(
                        inner_type, optional=True
                    )
                else:
                    decoders[field.name] = self._make_container_decoder(
                        inner_type, original_inner_type, optional=True
                    )
            else:
                decoders[field.name] = self._make_container_decoder(
                    field_type, original_type
                )

        decoder_func = partial(self.decode_dataclass, class_type, decoders)
        self.register_type(class_type, dc.asdict, decoder_func)

        return decoder_func

    def _make_scalar_decoder(
        self, field_type: Type, optional: bool = False
    ) -> Callable:
        if field_type in self._extra_types:
            decoder_func = self._extra_types[field_type]["decoder"]
        elif issubclass(field_type, enum.Enum):
            decoder_func = partial(self.decode_enum, field_type)
        elif dc.is_dataclass(field_type):
            decoder_func = self._make_dataclass_decoder(field_type)
        else:
            decoder_func = self._identity

        if optional:
            return lambda raw: None if raw is None else decoder_func(raw)
        else:
            return decoder_func

    def _make_container_decoder(
        self, field_type: Type, original_field_type: Type, optional: bool = False
    ) -> Callable:
        allowed_containers = {list, tuple, set, frozenset, dict}
        if original_field_type not in allowed_containers:
            raise TypeError(
                f"Unsupported type {field_type}, only the following container"
                f" types are allowed: {', '.join(str(t) for t in allowed_containers)}"
            )

        if original_field_type is dict:
            key_type, value_type = typing.get_args(field_type)
            key_decoder = self._make_scalar_decoder(self._unwrap_alias(key_type))
            value_decoder = self._make_scalar_decoder(self._unwrap_alias(value_type))

            decoder_func = partial(self.decode_dict, key_decoder, value_decoder)
        else:
            element_type = self._unwrap_alias(typing.get_args(field_type)[0])
            element_decoder = self._make_scalar_decoder(element_type)

            decoder_func = partial(
                self.decode_collection, original_field_type, element_decoder
            )

        if optional:
            return lambda raw: None if raw is None else decoder_func(raw)
        else:
            return decoder_func

    @staticmethod
    def _unwrap_alias(potential_alias: Type):
        if isinstance(potential_alias, FunctionType):
            if hasattr(potential_alias, "__supertype__"):
                # this is an alias created via typing.NewType()
                return potential_alias.__supertype__
            else:
                raise TypeError(
                    f"Type expected, found function instead: {potential_alias}"
                )

        return potential_alias

    @staticmethod
    def _identity(raw: Any) -> Any:
        return raw

    @staticmethod
    def decode_dataclass(
        class_type: Type, field_decoders: Dict[str, Callable], raw: dict
    ) -> Any:
        params = {}
        for field_name, decoder in field_decoders.items():
            params[field_name] = decoder(raw[field_name])

        return class_type(**params)

    @staticmethod
    def decode_enum(enum_type: Type, raw: Any) -> Any:
        return enum_type(raw)

    @staticmethod
    def decode_dict(key_decoder: Callable, value_decoder: Callable, raw: dict) -> dict:
        return {key_decoder(k): value_decoder(v) for k, v in raw.items()}

    @staticmethod
    def decode_collection(
        collection_factory: Callable, element_decoder: Callable, raw: Collection
    ) -> Collection:
        return collection_factory(element_decoder(e) for e in raw)


_default_serializer = RecordSerializer()

load_json = _default_serializer.load_json
load_dict = _default_serializer.load_dict
dump_json = _default_serializer.dump_json
register_type = _default_serializer.register_type
register_dataclass = _default_serializer.register_dataclass

