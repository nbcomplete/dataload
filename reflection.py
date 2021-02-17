import dataclasses as dc
from inspect import isabstract
from typing import (
    Any,
    ByteString,
    Collection,
    Generic,
    get_args,
    get_origin,
    Hashable,
    Mapping,
    Tuple,
    Type,
    TypeVar,
    Union,
)


T = TypeVar("T")


def is_real_collection(type_: type) -> bool:
    if not issubclass(type_, Collection):
        return False
    if issubclass(type_, (str, ByteString, memoryview, range)):
        return False
    return True


def is_dict_compatible(type_: type) -> bool:
    return issubclass(type_, Mapping) and not isabstract(type_)


def is_list_compatible(type_: type) -> bool:
    return (
        is_real_collection(type_)
        and not issubclass(type_, Mapping)
        and not isabstract(type_)
    )


def is_collection_constructible(target_type: type, source_type: type) -> bool:
    if issubclass(source_type, target_type) and not isabstract(target_type):
        return True
    if is_dict_compatible(target_type) and issubclass(source_type, Mapping):
        return True
    if is_list_compatible(target_type) and is_real_collection(source_type):
        return True

    return False


@dc.dataclass(frozen=True)
class TypeInfo(Generic[T]):
    real_type: Type[T]
    hinted_type: Type[T]
    type_args: Tuple["TypeInfo", ...] = ()
    is_optional: bool = False

    @property
    def is_real_collection(self):
        return is_real_collection(self.real_type)

    @property
    def is_mapping(self):
        return issubclass(self.real_type, Mapping)

    @property
    def is_generic(self):
        return len(self.type_args) > 0

    @property
    def key_type_info(self):
        if not self.is_mapping:
            raise TypeError

        try:
            kti = self.type_args[0]
        except IndexError:
            raise TypeError
        if not issubclass(kti.real_type, Hashable):
            raise TypeError

    @property
    def value_type_info(self):
        if not self.is_mapping:
            raise TypeError

        try:
            return self.type_args[1]
        except IndexError:
            raise TypeError

    @property
    def element_type_info(self):
        if not self.is_real_collection or self.is_mapping:
            raise TypeError

        try:
            return self.type_args[0]
        except IndexError:
            raise TypeError

    @property
    def is_dataclass(self):
        return dc.is_dataclass(self.real_type)

    def describes(self, obj: Any) -> bool:
        return isinstance(obj, self.real_type)

    def create(self, *args, **kwargs) -> T:
        return self.real_type(*args, **kwargs)

    def cast(self, value):
        return value if self.describes(value) else self.create(value)


def parse_typehint(hinted_type: Type[T]) -> TypeInfo[T]:
    real_type = get_origin(hinted_type)
    if not real_type:
        return TypeInfo(real_type=hinted_type, hinted_type=hinted_type)

    if real_type is Union:
        type_args = get_args(hinted_type)
        if len(type_args) == 2 and type_args[1] is type(None):
            return TypeInfo(
                real_type=type_args[0],
                hinted_type=hinted_type,
                is_optional=True,
            )
        else:
            raise TypeError

    return TypeInfo(
        real_type=real_type,
        hinted_type=hinted_type,
        type_args=tuple(parse_typehint(arg) for arg in get_args(hinted_type)),
    )
