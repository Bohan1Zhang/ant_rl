from __future__ import annotations

from typing import TypedDict, List


class AntState(TypedDict):
    x: int
    y: int
    colony: int
    carrying: int  # 0/1


class FoodState(TypedDict):
    x: int
    y: int
    amount: int


class NestState(TypedDict):
    x: int
    y: int
    colony: int


class Heatmap(TypedDict):
    w: int
    h: int
    data: List[int]  # flattened uint8


class WorldState(TypedDict):
    type: str  # "state"
    tick: int
    grid_w: int
    grid_h: int
    nests: List[NestState]
    ants: List[AntState]
    foods: List[FoodState]
    delivered: List[int]
    pheromone: Heatmap