import math
import time
import random
import tkinter
from NeuralNet import NeuralNet
from PIL import Image, ImageTk


# Individual constants:
# Bird neural net data:
INPUTS = 2
HIDDEN = [3]
OUTPUTS = 1
# Genetic algorithm constants:
MUTATION_PROB = 0.2
CROSSOVER_PROB = 0.5
MUTATION_MOVE_RANGE = 2
POPULATION_SIZE = 50
MAX_GENERATIONS = math.inf
TOURNAMENT_SIZE = 10
HALL_OF_FAME_SIZE = 5
# Game constants:
WINDOW_WIDTH = 900
WINDOW_HEIGHT = 600
FPS = 30
# Bird constants:
DEFAULT_X = 0
DEFAULT_Y = WINDOW_HEIGHT / 2
DEFAULT_SPEEDX = 400
DEFAULT_SPEEDY = 0
GRAVITY = 1600
JUMP_FORCE = -600
BIRD_WIDTH = 40
BIRD_HEIGHT = 30
# Walls constants:
WALL_BETWEEN = 150
WALL_WIDTH = 50
WALL_MIN_Y = WALL_BETWEEN
WALL_MAX_Y = WINDOW_HEIGHT - WALL_BETWEEN
# Camera constant:
CAMERA_DELTA_X = 100
# Constant, how fast show evolution
# ONLINE = True - birds learns online
# ONLINE = False - birds learns with maximum speed
ONLINE = False


class Camera:
    x: float
    y: float = 0

    def sync(self, obj):
        self.x = obj.x - CAMERA_DELTA_X

    def get_x(self, x: float):
        return x - self.x

    def get_y(self, y: float):
        return y - self.y


class HitBox:
    def __init__(self, center_coord: tuple, size: tuple):
        self.centerX, self.centerY = center_coord
        self.width, self.height = size

    def draw(self, color: str):
        canvas.create_rectangle(camera.get_x(self.centerX - self.width / 2),
                                camera.get_y(self.centerY - self.height / 2),
                                camera.get_x(self.centerX + self.width / 2),
                                camera.get_y(self.centerY + self.height / 2),
                                outline=color, width=2)

    def hasPoint(self, point: tuple):
        return abs(self.centerX - point[0]) < self.width / 2 and \
               abs(self.centerY - point[1]) < self.height / 2

    def hasStrike(self, other):
        return (
            self.hasPoint((other.centerX - other.width / 2, other.centerY - other.height / 2)) or
            self.hasPoint((other.centerX + other.width / 2, other.centerY - other.height / 2)) or
            self.hasPoint((other.centerX - other.width / 2, other.centerY + other.height / 2)) or
            self.hasPoint((other.centerX + other.width / 2, other.centerY + other.height / 2)) or
            other.hasPoint((self.centerX - self.width / 2, self.centerY - self.height / 2)) or
            other.hasPoint((self.centerX + self.width / 2, self.centerY - self.height / 2)) or
            other.hasPoint((self.centerX - self.width / 2, self.centerY + self.height / 2)) or
            other.hasPoint((self.centerX + self.width / 2, self.centerY + self.height / 2))
        )


class Wall:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.width = WALL_WIDTH
        self.top_hit_box = HitBox((self.x, self.y - WALL_BETWEEN / 2 - WINDOW_HEIGHT / 2), (WALL_WIDTH, WINDOW_HEIGHT))
        self.bottom_hit_box = HitBox((self.x, self.y + WALL_BETWEEN / 2 + WINDOW_HEIGHT / 2), (WALL_WIDTH, WINDOW_HEIGHT))

    def __repr__(self):
        return f"<Wall ({self.x}, {self.y})>"

    def draw(self):
        self.top_hit_box.draw('green')
        self.bottom_hit_box.draw('green')


class Bird:
    dead = False
    x = DEFAULT_X
    y = DEFAULT_Y
    speedX = DEFAULT_SPEEDX
    speedY = DEFAULT_SPEEDY
    width = BIRD_WIDTH
    height = BIRD_HEIGHT

    score: float = 0

    def __init__(self, genome = None):
        if genome is None:
            self.genome = NeuralNet(INPUTS, OUTPUTS, HIDDEN).json()['weights']
        else:
            self.genome = genome

    def mutate(self):
        for i in range(len(self.genome)):
            if random.random() < MUTATION_PROB:
                self.genome[i] += random.random() * 2 * MUTATION_MOVE_RANGE - MUTATION_MOVE_RANGE

    def crossover(self, other):
        for i in range(len(self.genome)):
            if random.random() < CROSSOVER_PROB:
                self.genome[i], other.genome[i] = other.genome[i], self.genome[i]

    @property
    def hit_box(self):
        return HitBox((self.x, self.y), (BIRD_WIDTH, BIRD_HEIGHT))

    @property
    def net(self):
        return NeuralNet.from_json({
            'inputs': INPUTS,
            'outputs': OUTPUTS,
            'hidden': HIDDEN,
            'weights': self.genome
        })

    def move(self, deltaTime):
        self.speedY += GRAVITY * deltaTime

        self.x += self.speedX * deltaTime
        self.y += self.speedY * deltaTime

        if self.y < 0 or self.y > WINDOW_HEIGHT:
            self.die()

        if not self.dead:
            self.score = self.x

    def jump(self, nearest_wall: Wall):
        distanceY = nearest_wall.y - self.y
        distanceX = nearest_wall.x + nearest_wall.width / 2 - self.x + self.width / 2
        distanceX /= WINDOW_WIDTH
        distanceY /= WINDOW_HEIGHT

        if self.net.push([distanceX, distanceY])[0] < 0.5:
            self.speedY = JUMP_FORCE

    def draw(self):
        self.hit_box.draw('red')

    def check_strike(self, wall: Wall):
        if self.hit_box.hasStrike(wall.top_hit_box) or self.hit_box.hasStrike(wall.bottom_hit_box):
            self.die()

    def die(self):
        if not self.dead:
            self.score = self.x
        self.dead = True

    def copy(self):
        return Bird(self.genome[:])


class Generation:
    walls: list[Wall] = []

    def __init__(self):
        self.population = []
        while len(self.population) < POPULATION_SIZE:
            self.population.append(Bird())

        self.hall_of_fame = []

    @property
    def _alive(self):
        return list(filter(lambda bird: not bird.dead,
                           self.population))

    @property
    def _best_bird(self):
        ans = self.population[0]
        for bird in self.population:
            if bird.score > ans.score:
                ans = bird
        return ans

    def _tournament(self):
        winner = random.choice(self.population)
        for i in range(TOURNAMENT_SIZE - 1):
            candidate = random.choice(self.population)
            if candidate.score > winner.score:
                winner = candidate
        return winner.copy()

    def _update_walls(self):
        if len(self.walls) == 0 or self.walls[-1].x < self._alive[0].x + WINDOW_WIDTH:
            if len(self.walls) == 0:
                x = WINDOW_HEIGHT
            else:
                x = self.walls[-1].x + WINDOW_HEIGHT

            self.walls.append(Wall(x, WALL_MIN_Y + random.random() * (WALL_MAX_Y - WALL_MIN_Y)))
        if self.walls[0].x < self._alive[0].x - WINDOW_WIDTH:
            self.walls = self.walls[1:]

    def change_population(self):
        self.hall_of_fame.append(self._best_bird)
        self.hall_of_fame = list(sorted(self.hall_of_fame,
                                        key=lambda bird: bird.score,
                                        reverse=True))
        self.hall_of_fame = self.hall_of_fame[:HALL_OF_FAME_SIZE]

        new_population = []
        while len(new_population) < POPULATION_SIZE - len(self.hall_of_fame):
            new_population.append(self._tournament())
        for bird1, bird2 in zip(new_population[::2], new_population[1::2]):
            bird1.crossover(bird2)
        for bird in new_population:
            bird.mutate()

        for bird in self.hall_of_fame:
            new_population.append(bird.copy())

        self.population = new_population
        self.walls = []

    def simulate_life(self):
        timer = time.time()
        while len(self._alive) > 0:
            canvas.delete('all')

            self._update_walls()

            deltaTime = time.time() - timer
            timer += deltaTime
            for bird in self.population:
                if ONLINE:
                    bird.move(deltaTime)
                else:
                    bird.move(1 / FPS)

            for bird in self.population:
                for wall in self.walls:
                    bird.check_strike(wall)

            nearest_wall = list(filter(
                lambda wall: wall.x + wall.width / 2 > self.population[0].x - self.population[0].width / 2,
                self.walls))[0]

            for bird in self.population:
                bird.jump(nearest_wall)

            camera.sync(self.population[0])
            for bird in self._alive:
                bird.draw()
            for wall in self.walls:
                wall.draw()

            canvas.create_text(150, 25, text=f'Generation: {generation_number}', font='Calibri 25')
            canvas.create_text(150, 50, text=f'Alive: {len(self._alive)}', font='Calibri 25')
            canvas.create_text(150, 75, text=f'Score: {int(self.get_best_score())}', font='Calibri 25')
            canvas.create_text(150, 100, text=f'Top score: {int(max(top_score, self.get_best_score()))}', font='Calibri 25')
            canvas.update()

    def get_best_score(self):
        return self._best_bird.score


if __name__ == "__main__":
    root = tkinter.Tk()
    root.title('Floppy Bird')
    canvas = tkinter.Canvas(root, width=WINDOW_WIDTH, height=WINDOW_HEIGHT)
    canvas.pack()
    camera = Camera()

    generation = Generation()
    generation_number = 1
    top_score = 0
    while generation_number <= MAX_GENERATIONS:
        print("Generation", generation_number)
        generation.simulate_life()
        top_score = max(top_score, generation.get_best_score())
        generation.change_population()
        generation_number += 1
