from string import ascii_letters
from tabulate import tabulate

import matplotlib.pyplot as plt
import random


CHROMOSOMES = ascii_letters + " "
MAX_CHARS = 20
MAX_POPULATION = 200
MAX_GENERATION = 10000


def generate_str(length: int) -> str:
    res = ""
    for _ in range(length):
        res += random.choice(CHROMOSOMES)
    return res


def valid_word(word: str) -> tuple[bool, str]:
    for char in word:
        if char not in CHROMOSOMES:
            return False, char
    return True, ""


class Environment:
    def __init__(self, population_size: int = MAX_POPULATION) -> None:
        self.pop_size = population_size
        self.get_target_word()
        self.init_population()

    def get_target_word(self) -> None:
        while True:
            self.target_word = input("Target Word: ")
            is_valid, invalid_char = valid_word(self.target_word)
            if not is_valid:
                print(f"[ERROR]: Invalid character: {invalid_char}\n")
            elif len(self.target_word) > MAX_CHARS:
                print(f"[ERROR]: Max. characters ({MAX_CHARS}) exceeded:",
                      end=" ")
                print(f"{len(self.target_word)}\n")
            else:
                break

    def init_population(self) -> None:
        self.population = []
        for _ in range(self.pop_size):
            self.population.append(generate_str(len(self.target_word)))



class GameMaster:
    def __init__(self, target_word: str) -> None:
        self.answer = target_word
        self.costs = []

    def calc_cost(self, guess: str) -> int:
        cost = 0
        for i in range(len(guess)):
            cost += (ord(guess[i]) - ord(self.answer[i])) ** 2
        return cost

    def show_result(self) -> None:
        plt.figure(figsize=(10, 5))
        plt.plot(self.costs, linestyle='-')
        plt.title('Convergence Graph')
        plt.xlabel('Generation')
        plt.ylabel('Cost Value')
        plt.grid(True)
        plt.show()

    def store_tally(
            self,
            tally: list[list[object]],
            filename: str = "result.txt"
    ) -> None:
        table_headers = ["Generation", "Guess", "Cost Value"]
        with open(filename, "w") as f:
            f.write(tabulate(tally, table_headers, "outline"))


class Guesser:
    def __init__(
            self,
            game_master: GameMaster,
            environment: Environment,
            num_generation: int = MAX_GENERATION
    ) -> None:
        self.num_generation = num_generation
        self.game_master = game_master
        self.env = environment

    def guess_word(self) -> None:
        current_gen = 1
        tally = []

        print("GUESSING THE WORD...")
        while current_gen <= self.num_generation:  # stopping technique
            # sort current population -> lowest to highest cost
            self.env.population = sorted(
                self.env.population, key=self.game_master.calc_cost
            )
            curr_best_cost = self.game_master.calc_cost(self.env.population[0])

            # store the current best
            self.game_master.costs.append(curr_best_cost)
            tally.append([current_gen, self.env.population[0], curr_best_cost])

            if curr_best_cost <= 0:  # stopping technique
                break

            # create new population
            self.regen_population()
            current_gen += 1

        result_filename = "word_guess.txt"
        self.game_master.store_tally(tally, result_filename)
        print(f"FINAL GUESS = {tally[-1][1]}")
        print(f"[DONE] Result is stored in file: {result_filename}")

        self.game_master.show_result()

    def generate_word(self, parent1: str, parent2: str) -> str:
        """Implements two-point crossover"""
        p = len(parent1) // 3
        children = [
            self.mutate(parent1[:p] + parent2[p:p*2] + parent1[p*2:]),
            self.mutate(parent2[:p] + parent1[p:p*2] + parent2[p*2:])
        ]
        best_child = sorted(children, key=self.game_master.calc_cost)

        return best_child[0]

    def mutate(self, word: str) -> str:
        """Picks two characters in the word and replace them with new ones"""
        rand_idx1 = random.randint(0, len(word) - 1)
        while True:
            rand_idx2 = random.randint(0, len(word) - 1)
            if rand_idx1 != rand_idx2:
                break
        while True:
            rand_char1 = random.choice(CHROMOSOMES)
            if rand_char1 != word[rand_idx1]: break
        while True:
            rand_char2 = random.choice(CHROMOSOMES)
            if rand_char2 != word[rand_idx2]: break

        word = word[:rand_idx1] + rand_char1 + word[rand_idx1 + 1:]
        word = word[:rand_idx2] + rand_char2 + word[rand_idx2 + 1:]

        return word

    def regen_population(self) -> None:
        new_gen = []
        new_gen.extend(self.env.population[:(10 * self.env.pop_size) // 100])
        for _ in range((90 * self.env.pop_size) // 100):
            parent1 = random.choice(
                self.env.population[:self.env.pop_size // 2]
            )
            while True:
                parent2 = random.choice(
                    self.env.population[:self.env.pop_size // 2]
                )
                if parent2 != parent1: break
            child = self.generate_word(parent1, parent2)
            new_gen.append(child)
        self.env.population = new_gen


def main() -> None:
    env = Environment()
    game_master = GameMaster(env.target_word)
    guesser = Guesser(game_master, env)
    guesser.guess_word()


if __name__ == "__main__":
    main()

