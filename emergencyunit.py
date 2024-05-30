from math import sqrt
from typing import Union, List
from tabulate import tabulate

import matplotlib.pyplot as plt
import random


MAX_GENERATION: int = 100
CITY: List[List[int]] = [
    [5, 2, 4, 8, 9, 0, 3, 3, 8, 7],
    [5, 5, 3, 4, 4, 6, 4, 1, 9, 1],
    [4, 1, 2, 1, 3, 8, 7, 8, 9, 1],
    [1, 7, 1, 6, 9, 3, 1, 9, 6, 9],
    [4, 7, 4, 9, 9, 8, 6, 5, 4, 2],
    [7, 5, 8, 2, 5, 2, 3, 9, 8, 2],
    [1, 4, 0, 6, 8, 4, 0, 1, 2, 1],
    [1, 5, 2, 1, 2, 8, 3, 3, 6, 2],
    [4, 5, 9, 6, 3, 9, 7, 6, 5, 10],
    [0, 6, 2, 8, 7, 1, 2, 1, 5, 3]
]


class Environment:
    def __init__(self, city_matrix: List[list[int]]) -> None:
        self.num_cities: int = len(city_matrix) * len(city_matrix[0])  # 10 x 10
        self.city_matrix: List[List[int]] = city_matrix
        # get all coords of every city and store as 1D list
        self.coords: List[tuple[int, int]] = self.get_coords()

    def get_coords(self) -> List[tuple[int, int]]:
        coords: List[tuple[int, int]] = []
        for y in range(len(self.city_matrix)):
            for x in range(len(self.city_matrix[y])):
                coords.append((x, y))
        return coords


class Locator:
    def __init__(self, env: Environment) -> None:
        self.env: Environment = env
        self.costs: List[float] = []

    def find_best_loc(self, max_gen: int = MAX_GENERATION) -> None:
        parents: List[tuple[int, int]] = [random.choice(self.env.coords),
                                          random.choice(self.env.coords)]

        tally: List[List[Union[tuple[int, int], float, int]]] = []
        # content will be [generation, coord, cost, fire freq]
        curr_best: List[Union[tuple[int, int], float, int]] = []
        curr_gen = 1
        while curr_gen <= max_gen:
            best_parent: tuple[int, int] = self.get_best_indv(parents)
            new_offspring: tuple[int, int] = self.gen_offspring(parents[0],
                                                                parents[1])
            offspring_cost: float = self.calc_cost(new_offspring)

            if offspring_cost > self.calc_cost(best_parent):
                freq = self.env.city_matrix[new_offspring[1]][new_offspring[0]]
                curr_best = [new_offspring, offspring_cost, freq]
                parents[0] = new_offspring
                while True:
                    # continuously generate new 2nd parent until its unique
                    # with new 1st parent (the best offspring so far)
                    new_parent = random.choice(self.env.coords)
                    if not self.is_same_loc(new_offspring, new_parent):
                        parents[1] = new_parent
                        break
            else:
                parents[1] = random.choice(self.env.coords)
            if len(curr_best) > 0:
                tally.append([curr_gen] + curr_best)
                curr_gen += 1
                self.costs.append(curr_best[1]) # type: ignore

        self.store_tally(tally, "emergency_unit.txt") # type: ignore
        self.show_result()

    def get_best_indv(self, coords: list[tuple[int, int]]) -> tuple[int, int]:
        costs = []
        for i in range(len(coords)):
            costs.append((self.calc_cost(coords[i]), i))
        best = sorted(costs, key=lambda c: c[0], reverse=True)[0]
        return coords[best[1]]

    def calc_cost(self, proposed_loc: tuple[int, int]) -> float:
        cost = 0
        for coord in self.env.coords:
            xn, yn = coord; xfs, yfs = proposed_loc
            fire_freq = self.env.city_matrix[yn][xn]
            cost += fire_freq * sqrt((xn - xfs) ** 2 + (yn - yfs) ** 2)
        return cost

    def gen_offspring(
            self,
            p1: tuple[int, int],
            p2: tuple[int, int]
    ) -> tuple[int, int]:
        return self.mutate(self.get_best_indv([(p1[0], p2[1]), (p2[0], p1[1])]))

    def mutate(self, offspring: tuple[int, int]) -> tuple[int, int]:
        coord_to_mutate = random.choice([0, 1])
        if (coord_to_mutate == 0):
            return (random.randint(0, 9), offspring[1])
        return (offspring[0], random.randint(0, 9))

    def is_same_loc(
            self,
            coord1: tuple[int, int],
            coord2: tuple[int, int]
    ) -> bool:
        if coord1[0] == coord2[0] and coord1[1] == coord2[1]:
            return True
        return False

    def show_result(self) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(self.costs, marker="o", linestyle='-')
        plt.title('Convergence Graph')
        plt.xlabel('Generation')
        plt.ylabel('Cost Value')
        plt.grid(True)
        plt.show()

    def store_tally(
            self,
            tally: list[Union[tuple[int, int], float, int]],
            filename: str = "result.txt"
    ) -> None:
        table_headers = ["Generation", "Guess", "Cost Value", "Frequency"]
        with open(filename, "w") as f:
            f.write(tabulate(tally, table_headers, "outline")) # type: ignore


def main() -> None:
    env = Environment(CITY)
    locator = Locator(env)
    locator.find_best_loc()


if __name__ == "__main__":
    main()

