import dataclasses as dc
import datetime as dt
import enum
import json
from decimal import Decimal
from functools import partial
from types import FunctionType
from typing import Any, Callable, Collection, Dict, List, Type, Union
from uuid import UUID

import typing


__all__ = [
    "RecordSerializer",
    "SerializerTypeError",
    "load_dict",
    "load_json",
    "register_dataclass",
    "register_type",
    "serialize_to_dict",
    "serialize_to_json",
]

from simple.json_format import (
    datetime_to_json,
    json_to_datetime,
    json_to_time,
    time_to_json,
)


class SerializerTypeError(TypeError):
    pass


class RecordSerializer:
    _ALLOWED_CONTAINERS = frozenset({list, tuple, set, frozenset, dict})

    def __init__(self):
        self._extra_types: Dict[Type, Dict[str, Callable]] = {}
        self.register_type(UUID, encoder=str, decoder=UUID)
        self.register_type(Decimal, encoder=str, decoder=Decimal)
        self.register_type(
            dt.datetime, encoder=datetime_to_json, decoder=json_to_datetime
        )
        self.register_type(dt.time, encoder=time_to_json, decoder=json_to_time)
        self.register_type(
            dt.date,
            encoder=lambda d: d.isoformat(),
            decoder=lambda s: dt.date.fromisoformat(s),
        )

    def register_type(self, type_: Type, encoder: Callable, decoder: Callable):
        self._extra_types[type_] = {
            "encoder": encoder,
            "decoder": decoder,
        }

    def register_dataclass(self, class_type: Type):
        if not dc.is_dataclass(class_type):
            raise SerializerTypeError(f"{class_type} is not a dataclass")

        constructed: Dict[Type, Dict[str, Callable]] = {}
        self._make_dataclass_decoder(class_type, constructed)
        for type_, entry in constructed.items():
            self.register_type(type_, entry["encoder"], entry["decoder"])

    def serialize_to_json(self, obj: Any, **kwargs) -> str:
        return json.dumps(self.serialize_to_dict(obj), **kwargs)

    def serialize_to_dict(self, obj: Any) -> Dict[str, Any]:
        if not dc.is_dataclass(obj):
            raise SerializerTypeError(
                f"Cannot serialize type {type(obj)}, not a dataclass"
            )

        return self._encode_dataclass(obj)

    def load_json(self, class_type: Type, raw: str, **kwargs) -> Any:
        return self.load_dict(class_type, json.loads(raw, **kwargs))

    def load_dict(self, class_type: Type, raw: Dict[str, Any]) -> Any:
        if not dc.is_dataclass(class_type):
            raise SerializerTypeError(
                f"Cannot decode type {class_type}, not a dataclass"
            )

        if class_type not in self._extra_types:
            self.register_dataclass(class_type)

        return self._extra_types[class_type]["decoder"](raw)

    def _encode_dataclass(self, obj: Any) -> Dict[str, Any]:
        result = {}
        field: dc.Field
        for field in dc.fields(obj):
            result[field.name] = self._encode_value(getattr(obj, field.name))

        return result

    def _encode_value(self, obj: Any) -> Any:
        type_ = type(obj)
        if type_ in self._extra_types:
            return self._extra_types[type(obj)]["encoder"](obj)
        elif issubclass(type_, enum.Enum):
            return obj.value
        elif type_ is dict:
            return self._encode_dict(obj)
        elif type_ in self._ALLOWED_CONTAINERS:
            return self._encode_collection(obj)
        elif dc.is_dataclass(type_):
            return self._encode_dataclass(obj)
        else:
            return obj

    def _encode_dict(self, obj: Dict[Any, Any]) -> Dict[Any, Any]:
        return {self._encode_value(k): self._encode_value(v) for k, v in obj.items()}

    def _encode_collection(self, obj: Collection[Any]) -> List[Any]:
        return [self._encode_value(e) for e in obj]

    def _make_dataclass_decoder(
        self, class_type: Type, in_construction: Dict[Type, Dict[str, Callable]]
    ):
        class_type = self._unwrap_alias(class_type)
        if class_type in self._extra_types:
            return self._extra_types[class_type]["decoder"]
        elif class_type in in_construction:
            # We already encountered this type higher up in the field tree, just refer
            # to the decoder func to avoid an infinite recursion
            return in_construction[class_type]["decoder"]

        # We need to construct and keep track of the decoder func first,
        # so that we can refer to it in case we encounter a recursive reference
        # down the line (e.g. a class `Person` having a field `friends` of type `List[Person]`)
        field_decoders = {}
        decoder_func = partial(self.decode_dataclass, class_type, field_decoders)
        in_construction[class_type] = {
            "encoder": self._encode_dataclass,
            "decoder": decoder_func,
        }

        for field_name, field_type in typing.get_type_hints(class_type).items():
            field_type = self._unwrap_alias(field_type)
            original_type = self._unwrap_alias(typing.get_origin(field_type))
            if original_type is None:
                if field_type in self._ALLOWED_CONTAINERS:
                    raise SerializerTypeError(
                        f"Cannot decode type {field_type}, container type hints have"
                        f" to include their element types"
                    )
                field_decoders[field_name] = self._make_scalar_decoder(
                    field_type, in_construction
                )
            elif original_type is Union:
                type_args = typing.get_args(field_type)
                if len(type_args) != 2 or type_args[1] is not type(None):
                    raise SerializerTypeError(
                        f"Cannot decode type {field_type}, Union is not allowed unless"
                        f" it represents Optional[T]"
                    )
                inner_type = self._unwrap_alias(type_args[0])
                original_inner_type = typing.get_origin(inner_type)
                if original_inner_type is None:
                    field_decoders[field_name] = self._make_scalar_decoder(
                        inner_type, in_construction, optional=True
                    )
                else:
                    field_decoders[field_name] = self._make_container_decoder(
                        inner_type, original_inner_type, in_construction, optional=True
                    )
            else:
                field_decoders[field_name] = self._make_container_decoder(
                    field_type, original_type, in_construction
                )

        return decoder_func

    def _make_scalar_decoder(
        self,
        field_type: Type,
        in_construction: Dict[Type, Dict[str, Callable]],
        optional: bool = False,
    ) -> Callable:
        if field_type in self._extra_types:
            decoder_func = self._extra_types[field_type]["decoder"]
        elif issubclass(field_type, enum.Enum):
            decoder_func = partial(self.decode_enum, field_type)
        elif dc.is_dataclass(field_type):
            decoder_func = self._make_dataclass_decoder(field_type, in_construction)
        else:
            decoder_func = self._identity

        if optional:
            return lambda raw: None if raw is None else decoder_func(raw)
        else:
            return decoder_func

    def _make_container_decoder(
        self,
        field_type: Type,
        original_field_type: Type,
        in_construction: Dict[Type, Dict[str, Callable]],
        optional: bool = False,
    ) -> Callable:
        if original_field_type not in self._ALLOWED_CONTAINERS:
            raise SerializerTypeError(
                f"Cannot decode type {field_type}, only the following container"
                f" types are allowed: {', '.join(str(t) for t in self._ALLOWED_CONTAINERS)}"
            )

        if original_field_type is dict:
            key_type, value_type = typing.get_args(field_type)
            key_decoder = self._make_scalar_decoder(
                self._unwrap_alias(key_type), in_construction
            )
            value_decoder = self._make_scalar_decoder(
                self._unwrap_alias(value_type), in_construction
            )

            decoder_func = partial(self.decode_dict, key_decoder, value_decoder)
        else:
            element_type = self._unwrap_alias(typing.get_args(field_type)[0])
            element_decoder = self._make_scalar_decoder(element_type, in_construction)

            decoder_func = partial(
                self.decode_collection, original_field_type, element_decoder
            )

        if optional:
            return lambda raw: None if raw is None else decoder_func(raw)
        else:
            return decoder_func

    @staticmethod
    def _unwrap_alias(potential_alias: Union[Type, FunctionType]) -> Type:
        if isinstance(potential_alias, FunctionType):
            if hasattr(potential_alias, "__supertype__"):
                # this is an alias created via typing.NewType()
                return potential_alias.__supertype__
            else:
                raise SerializerTypeError(
                    f"Type expected, found function instead: {potential_alias}"
                )
        else:
            # Nothing to unwrap
            return potential_alias

    @staticmethod
    def _identity(raw: Any) -> Any:
        return raw

    @staticmethod
    def decode_dataclass(
        class_type: Type, field_decoders: Dict[str, Callable], raw: Dict[str, Any]
    ) -> Any:
        params = {}
        for field_name, decoder in field_decoders.items():
            params[field_name] = decoder(raw[field_name])

        return class_type(**params)

    @staticmethod
    def decode_enum(enum_type: Type, raw: Any) -> Any:
        return enum_type(raw)

    @staticmethod
    def decode_dict(
        key_decoder: Callable, value_decoder: Callable, raw: Dict[Any, Any]
    ) -> Dict[Any, Any]:
        return {key_decoder(k): value_decoder(v) for k, v in raw.items()}

    @staticmethod
    def decode_collection(
        collection_factory: Callable, element_decoder: Callable, raw: Collection[Any]
    ) -> Collection[Any]:
        return collection_factory(element_decoder(e) for e in raw)


_default_serializer = RecordSerializer()

serialize_to_json = _default_serializer.serialize_to_json
serialize_to_dict = _default_serializer.serialize_to_dict
load_json = _default_serializer.load_json
load_dict = _default_serializer.load_dict
register_type = _default_serializer.register_type
register_dataclass = _default_serializer.register_dataclass

