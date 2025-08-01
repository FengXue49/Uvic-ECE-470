import pygame  # Import "Pygame" for pygame interface
import heapq  # Import heapq to achieve A* algorithm priority queue
import random  # for random choose, crossover and mutation in genetic algorithm
import time # to show time
from collections import defaultdict # import default dictionary for storing g_score

# ================= Initialize Pygame =================
pygame.init()
WIDTH, HEIGHT = 800, 650
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Path Finder")

# ================= Color definition =================
COLORS_ORDER = ["red", "blue", "green", "orange", "purple"]
COLORS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 201, 87),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "gray": (200, 200, 200)
}

# ================= Grid Parameters =================
GRID_SIZE = 10
DOT_SPACING = 50
GRID_START_X = 200
GRID_START_Y = 50

# ================= UI Parameters =================
COLOR_BUTTON_SIZE = 40
START_BUTTON = pygame.Rect(200, HEIGHT - 50, 100, 30)
RESET_BUTTON = pygame.Rect(320, HEIGHT - 50, 100, 30)

# ================= Data Storage =================
points = {c: [] for c in COLORS_ORDER}  # List of endpoints for each color, Max 2 each - For each color c in COLORS_ORDER, create a key-value pair c: [] in the dictionary
selected_color = None  # Current Selecting Color
solution = {}          # Path dictionary storing the final solution; Color -> Path
solution_computed = False  # Determine if a solution is computed
color_placement_order = []  # Record the order of colors user placed on the grid

# ================= Drawing Functions =================
def draw_grid():
    "Draw the grid dots"
    for grid_x in range(GRID_SIZE):
        for grid_y in range(GRID_SIZE):
            center_x = GRID_START_X + grid_x * DOT_SPACING
            center_y = GRID_START_Y + grid_y * DOT_SPACING
            pygame.draw.circle(screen, COLORS['black'], (center_x, center_y), 3)


def draw_color_panel():
    "Draw the left Color Panel"
    panel_y = 50 # Vertical distance
    for color in COLORS_ORDER:
        rect = pygame.Rect(20, panel_y, COLOR_BUTTON_SIZE, COLOR_BUTTON_SIZE)
        pygame.draw.rect(screen, COLORS[color], rect)
        if len(points[color]) == 2:
            pygame.draw.rect(screen, COLORS['black'], rect, 3)
        if selected_color == color:
            pygame.draw.rect(screen, COLORS['gray'], rect, 3)
        panel_y += COLOR_BUTTON_SIZE + 10

# ================= Line intersection Detection =================
def lines_intersect(point1, point2, query_point1, query_point2):
    "Determine whether two lines intersect"
    # Vetcor Corss product
    def orient(point_a, point_b, point_c):
        # AB×AC=(xb−xa)(yc−ya)−(yb−ya)(xc−xa)
        val = (point_b[0]-point_a[0])*(point_c[1]-point_a[1]) - (point_b[1]-point_a[1])*(point_c[0]-point_a[0])
        if abs(val) < 1e-9:  # Collinear
            return 0
        return 1 if val > 0 else -1 
        # val > 0: c is at the left side of a->b, counter-clockwise
        # val < 0: c is at the right side of a->b, clockwise
        # val = 0: collinear
    
    orientation_1 = orient(point1, point2, query_point1)
    orientation_2 = orient(point1, point2, query_point2)
    orientation_3 = orient(query_point1, query_point2, point1)
    orientation_4 = orient(query_point1, query_point2, point2)
    # q1 -> p1p2 
    # q2 -> p1p2
    # p1 -> q1q2
    # p2 -> q1q2
    
    # General case: lines intersect
    # if q1 and q2 are at the different side of p1p2
    if orientation_1 != orientation_2 and orientation_3 != orientation_4:
        return True
    
    # Special case: collinear and overlapping
    # Determine if the point_b is between point_a and point_c
    def on_segment(point_a, point_b, point_c):
        # min(xa, xc) <= xb <= max(xa, xc)
        # min(ya, yc) <= yb <= max(ya, yc)
        return (min(point_a[0], point_c[0]) <= point_b[0] <= max(point_a[0], point_c[0]) and 
                min(point_a[1], point_c[1]) <= point_b[1] <= max(point_a[1], point_c[1]))
    
    # Only if the two lines are collinear and point is between two points --> Overlapping
    if (orientation_1 == 0 and on_segment(point1, query_point1, point2)) or \
       (orientation_2 == 0 and on_segment(point1, query_point2, point2)) or \
       (orientation_3 == 0 and on_segment(query_point1, point1, query_point2)) or \
       (orientation_4 == 0 and on_segment(query_point1, point2, query_point2)):
        return True
    
    return False

def paths_intersect(path1, path2):
    # If either path is empty, return False
    if not path1 or not path2:
        return False
    
    # path is a list consist with dots (x1, y1), (x2, y2)...
    # Check if each segment is intersecting with another segment
    for segment_i in range(len(path1) - 1):
        for segment_j in range(len(path2) - 1):
            if lines_intersect(path1[segment_i], path1[segment_i+1], path2[segment_j], path2[segment_j+1]):
                return True
    return False

# ================= A* Path Finder =================
class GridPathFinder:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.obstacles = set()

    def update_obstacles(self, connections):
        "Convert existing paths to obstacles"
        self.obstacles.clear()
        for path in connections.values():
            if path:  # Make sure the path exists
                for segment_i in range(len(path) - 1):
                    self._add_line_obstacles(path[segment_i], path[segment_i+1])

    def _add_line_obstacles(self, start, end):
        "Apply Bresenham Algorithm to add obstacle points on the line"
        start_grid_x, start_grid_y = self._screen_to_grid(start)
        end_grid_x, end_grid_y = self._screen_to_grid(end)
        
        delta_x = abs(end_grid_x - start_grid_x)
        delta_y = abs(end_grid_y - start_grid_y)
        step_x = 1 if start_grid_x < end_grid_x else -1 # 1 right, -1 left
        step_y = 1 if start_grid_y < end_grid_y else -1 # 1 up, -1 down
        error = delta_x - delta_y # Initial error = horizontal distance - vertical distance
        
        current_x, current_y = start_grid_x, start_grid_y
        
        while True:
            self.obstacles.add((current_x, current_y))
            if current_x == end_grid_x and current_y == end_grid_y:
                break
            error_double = 2 * error
            if error_double > -delta_y: # update x
                error -= delta_y
                current_x += step_x
            if error_double < delta_x: # update y
                error += delta_x
                current_y += step_y

    def _screen_to_grid(self, pos):
        # pos[mouse_x, mouse_y]
        "Screen Coordinate convert to Grid Coordinate"
        grid_x = round((pos[0] - GRID_START_X) / DOT_SPACING)
        grid_y = round((pos[1] - GRID_START_Y) / DOT_SPACING)
        # Constrain the pos to be within the grid
        return max(0, min(self.grid_size - 1, grid_x)), max(0, min(self.grid_size - 1, grid_y))

    def find_path(self, start, end, existing_connections):
        "Apply A* Alogorithm to find path"
        self.update_obstacles(existing_connections)

        # Convert the screen coordinate to grid coordinate
        start_grid_x, start_grid_y = self._screen_to_grid(start)
        end_grid_x, end_grid_y = self._screen_to_grid(end)
        
        # Add endpoints of other colors as obstacle
        # points.item() returns a [('color1', [(x1, y1)]), ('color2', [(x2, y2)])]
        for color, point_list in points.items():
            for point in point_list:
                point_grid_x, point_grid_y = self._screen_to_grid(point)
                if (point_grid_x, point_grid_y) not in [(start_grid_x, start_grid_y), (end_grid_x, end_grid_y)]:
                    self.obstacles.add((point_grid_x, point_grid_y))
        
        # A* algorithm
        # open_set = [unexplored nodes]
        open_set = []
        heapq.heappush(open_set, (0, (start_grid_x, start_grid_y))) # add new node
        came_from = {}
        g_score = defaultdict(lambda: float('inf'))
        g_score[(start_grid_x, start_grid_y)] = 0
        f_score = defaultdict(lambda: float('inf'))
        f_score[(start_grid_x, start_grid_y)] = abs(start_grid_x - end_grid_x) + abs(start_grid_y - end_grid_y)
        # f_score = g_score + h_score
        # h_score = abs(x1 - x2) + abs(y1 - y2)
        
        while open_set:
            # Take out the node with smallest f_score
            _, current = heapq.heappop(open_set) # _ don't care about the priority
            # current -> node(x, y)
            current_x, current_y = current
            # Get previous point coordinate
            # prev = came_from.get((current_x, current_y), None)
            
            if (current_x, current_y) == (end_grid_x, end_grid_y):
                # Re-construct the path
                path = []
                # while current position still has a parent, keep in loop
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append((start_grid_x, start_grid_y))
                # reverse the path: End -> Start => Start -> End
                path.reverse() 
                return self._path_to_screen(path)
            
            # Check the neighbor points - 8 directions
            for delta_x in (-1, 0, 1):
                for delta_y in (-1, 0, 1):
                    if delta_x == 0 and delta_y == 0:
                        continue
                    
                    # explore the neighbor points
                    neighbor_x = current_x + delta_x
                    neighbor_y = current_y + delta_y
                    
                    # Check the boundary
                    if not (0 <= neighbor_x < self.grid_size and 0 <= neighbor_y < self.grid_size):
                        continue
                    
                    # Check the obstacle
                    if (neighbor_x, neighbor_y) in self.obstacles:
                        continue

                    # if prev and abs(prev[0] - neighbor_x) + abs(prev[1] - neighbor_y) > 1:
                    #     continue
                    
                    # Calculate the movement cost
                    move_cost = 2 if abs(delta_x) == 1 and abs(delta_y) == 1 else 1.0
                    # Starting point to current g_score
                    tentative_g = g_score[(current_x, current_y)] + move_cost
                    
                    if tentative_g < g_score[(neighbor_x, neighbor_y)]:
                        came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)
                        g_score[(neighbor_x, neighbor_y)] = tentative_g
                        f_score[(neighbor_x, neighbor_y)] = tentative_g + abs(neighbor_x - end_grid_x) + abs(neighbor_y - end_grid_y)
                        heapq.heappush(open_set, (f_score[(neighbor_x, neighbor_y)], (neighbor_x, neighbor_y)))
        
        return None  # Can't find a path

    def _path_to_screen(self, grid_path):
        "Grid Coordinate convert to Screen Coordinate"
        return [(GRID_START_X + grid_x * DOT_SPACING, GRID_START_Y + grid_y * DOT_SPACING) 
                for grid_x, grid_y in grid_path]

path_finder = GridPathFinder(GRID_SIZE)

# ================= Algorithm for connecting in the order of user placement =================
def user_order_connect():
    "Connect the lines based on the order of user placement"
    if not color_placement_order:
        return {}
    
    print(f"Based on user placement order: {color_placement_order}")
    
    user_order_solution = {}
    for color in color_placement_order:
        if len(points[color]) == 2:
            point1, point2 = points[color] # Assign starting point and ending point to "point1" and "point2"
            path = path_finder.find_path(point1, point2, user_order_solution) # Sart to find path
            
            if path is not None: # Solution exists
                # Check the intersection with existing lines
                has_intersection = False
                for existing_path in user_order_solution.values():
                    if paths_intersect(path, existing_path):
                        has_intersection = True
                        break
                
                if not has_intersection: # If True (No intersection): Valid
                    user_order_solution[color] = path
                    print(f"User order connected color: {color}")
                else:
                    print(f"Color {color} intersects with other lines, skip")
            else:
                print(f"Color {color} can't find a path, skip")
    
    return user_order_solution

def priority_connect():
    "Connect based on the priority"
    PRIORITY_ORDER = ["red", "blue", "green", "orange", "purple"]
    
    valid_colors = [c for c in PRIORITY_ORDER if len(points[c]) == 2]
    if not valid_colors:
        return {}
    
    print(f"Connect based on the priority: {valid_colors}")
    
    priority_solution = {}
    for color in valid_colors:
        point1, point2 = points[color]
        path = path_finder.find_path(point1, point2, priority_solution)
        
        if path is not None:
            # Check intersection with exsisting lines
            has_intersection = False
            for existing_path in priority_solution.values():
                if paths_intersect(path, existing_path):
                    has_intersection = True
                    break
            
            if not has_intersection:
                priority_solution[color] = path
                print(f"priority algorithm connected: {color}")
            else:
                print(f"Color {color} in intersecting with existing line, skip")
        else:
            print(f"Color {color} can't find a path, skip")
    
    return priority_solution

def greedy_connect():
    "greedy algorithm: connect easier path first"
    valid_colors = [c for c in COLORS_ORDER if len(points[c]) == 2]
    if not valid_colors:
        return {}
    
    print("Trying greedy algorithm")
    
    # Calculate the path difficulty for each color (Manhattan distance)
    color_distances = []
    for color in valid_colors:
        point1, point2 = points[color]
        # Conver to grid coordinate
        grid_x1, grid_y1 = path_finder._screen_to_grid(point1)
        grid_x2, grid_y2 = path_finder._screen_to_grid(point2)
        # Find the Manhattan distance
        distance = abs(grid_x1 - grid_x2) + abs(grid_y1 - grid_y2)
        color_distances.append((distance, color))
    
    # Sort according to distance, connect shorter path first
    color_distances.sort()
    
    greedy_solution = {}
    for distance, color in color_distances:
        point1, point2 = points[color]
        path = path_finder.find_path(point1, point2, greedy_solution)
        
        if path is not None:
            # Check intersection with exsisting lines
            has_intersection = False
            for existing_path in greedy_solution.values():
                if paths_intersect(path, existing_path):
                    has_intersection = True
                    break
            
            if not has_intersection:
                greedy_solution[color] = path
                print(f"greedy algorithm connected: {color}")
    
    return greedy_solution

# ================= Genetic Algorithm functions =================
POP_SIZE = 50 # Population Size
GEN_MAX = 100 # Max Generation
CROSS_RATE = 0.8 # Crossover Rate
MUT_RATE = 0.3 # Mutation Rate

def random_individual():
    "Generate random individuals (Color order) 5! = 120"
    valid_colors = [c for c in COLORS_ORDER if len(points[c]) == 2]
    if not valid_colors:
        return []
    individual = valid_colors.copy()
    random.shuffle(individual)
    return individual

def evaluate(individual):
    "Evaluate individual fitness"
    if not individual:
        return -1e6, {} # Extreme low fitness and empty solution dictionary
    
    temp_solution = {}
    total_length = 0
    connected_count = 0
    
    # Try to connect based on individual order, skip the color can't be connected
    for color in individual:
        if len(points[color]) < 2:
            continue
            
        point1, point2 = points[color]
        path = path_finder.find_path(point1, point2, temp_solution)
        
        if path is None:
            # Can't find path, skip this color
            continue
        
        # Check intersection with exsisting lines
        has_intersection = False
        for existing_color, existing_path in temp_solution.items():
            if paths_intersect(path, existing_path):
                has_intersection = True
                break
        
        # If no intersection, add to solution
        if not has_intersection:
            # Calculate the path length
            path_length = 0
            for segment_i in range(len(path) - 1):
                # path: [(200,50),(250,50),(300,100),...]
                delta_x = path[segment_i][0] - path[segment_i+1][0]
                delta_y = path[segment_i][1] - path[segment_i+1][1]
                # Find the total length of the found path
                # Euclidean distance
                path_length += (delta_x*delta_x + delta_y*delta_y)**0.5
            total_length += path_length
            
            temp_solution[color] = path
            connected_count += 1
        # If there is intersection, skip this color, and try next one
        # else:
        #     print(f"Color {color} can't find a path, skip")
        #     break

    # fitness = connected count * 1000 - total length
    # Prioritize more color connected then consider path length
    fitness = connected_count * 1000 - total_length
    return fitness, temp_solution

def select_parent(population, fitness_scores):
    "Roulette Selection"
    # Convert negative fitness to postive value
    # fitness: (fitness, solution)
    # Take out the smallest one and conver to 0 (fitness - min_fitness)
    # +1: So that even all firness are 0s, at least they are 1 not 0
    min_fitness = min(fitness for fitness, _ in fitness_scores)
    adjusted_fitness = [fitness - min_fitness + 1 for fitness, _ in fitness_scores]
    
    total_fitness = sum(adjusted_fitness)

    # If all fitness less or equal to 0, randomly pick
    # Since we adjusted the fitness to be at least 1, this is just in case
    if total_fitness <= 0:
        return random.choice(population)
    
    # Spin the roulette wheel
    pick = random.uniform(0, total_fitness)

    current = 0
    for individual_index, fitness in enumerate(adjusted_fitness):
        current += fitness
        # |——f0——|——f1——|——f2——|...
        # if f0 ≤ pick < f0 + f1, return f1 as parent
        if current >= pick:
            return population[individual_index]
    
    # Just in case if the code didn't return an individual
    return population[-1]

def crossover(parent1, parent2):
    "Crossover: Receive two parents"

    # Randomly skip the crossover and return a copy of parent1
    if random.random() > CROSS_RATE or len(parent1) <= 1:
        return parent1[:]
    
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    
    child = [None] * size
    child[start:end+1] = parent1[start:end+1]
    
    # Pointing to the next position to be filled
    pointer = (end + 1) % size

    for item in parent2[end+1:] + parent2[:end+1]:
        if item not in child:
            child[pointer] = item
            pointer = (pointer + 1) % size
    
    return child

def mutate(individual):
    "Mutation (Swap two position)"
    if random.random() < MUT_RATE and len(individual) > 1:
        index_i, index_j = random.sample(range(len(individual)), 2)
        individual[index_i], individual[index_j] = individual[index_j], individual[index_i]

def run_genetic_algorithm():
    "run genetic algorithm"

    # Check if there are two points for each color
    valid_colors = [c for c in COLORS_ORDER if len(points[c]) == 2]
    if len(valid_colors) == 0:
        print("Please place two points for each color.")
        return {}
    
    print(f"Starting the genetic algorithm, color need to be connected: {valid_colors}")
    
    # Initilize the population
    population = [random_individual() for _ in range(POP_SIZE)]
    best_solution = {}
    best_fitness = -1e9
    best_connected_count = 0
    
    for generation in range(GEN_MAX):
        # Fitness evaluation
        fitness_scores = [evaluate(individual) for individual in population]
        
        # Update the best solution
        for individual_index, (fitness, solution) in enumerate(fitness_scores):
            connected_count = len(solution)
            # if fitness is the same, compare the connected count
            if fitness > best_fitness or (fitness == best_fitness and connected_count > best_connected_count):
                best_fitness = fitness
                best_solution = solution.copy()
                best_connected_count = connected_count
                print(f"Generation #{generation} found a better solution, connected count: {connected_count}, fitness: {fitness:.2f}")
        
        # End if all connected
        if best_connected_count == len(valid_colors):
            print(f"Found the perfect solution, connected all{best_connected_count}color!")
            break
        
        # Generate the new generation
        new_population = []
        
        # Loop POP_SIZE times
        for _ in range(POP_SIZE):
            parent1 = select_parent(population, fitness_scores)
            parent2 = select_parent(population, fitness_scores)
            child = crossover(parent1, parent2)
            mutate(child)
            new_population.append(child)
        
        population = new_population
    
    print(f"Genetic algorithm is completed, find connected count is: {best_connected_count}, fitness: {best_fitness:.2f}")
    print(f"the color connected are: {list(best_solution.keys())}")
    
    # Display unconnected color
    unconnected = [c for c in valid_colors if c not in best_solution]
    if unconnected:
        print(f"The color that are not able to connected: {unconnected}")
    
    return best_solution

# ================= Print out the order of colors user placed =================
def add_color_to_placement_order(color):
    "Add color into the placement oorder list"
    if color not in color_placement_order:
        color_placement_order.append(color)
        print(f"Color {color} is added into the placement order, current order is: {color_placement_order}")

# ================= Main Loop =================
running = True
clock = pygame.time.Clock()

while running:
    screen.fill(COLORS['white'])
    
    # Draw the UI
    draw_color_panel()
    draw_grid()
    
    # Draw the button
    pygame.draw.rect(screen, COLORS['gray'], START_BUTTON)
    pygame.draw.rect(screen, COLORS['gray'], RESET_BUTTON)
    
    font = pygame.font.SysFont(None, 24)
    screen.blit(font.render('Start', True, COLORS['black']), 
                (START_BUTTON.x + 20, START_BUTTON.y + 5))
    screen.blit(font.render('Reset', True, COLORS['black']), 
                (RESET_BUTTON.x + 20, RESET_BUTTON.y + 5))
    
    # Print the current placement order
    if color_placement_order:
        order_text = f"Order: {' -> '.join(color_placement_order)}"
        order_surface = font.render(order_text, True, COLORS['black'])
        screen.blit(order_surface, (200, 20))
    
    # Draw the dots
    for color in COLORS_ORDER:
        for point in points[color]:
            pygame.draw.circle(screen, COLORS[color], point, 8)
    
    # Draw the path
    if solution_computed and solution:
        for color, path in solution.items():
            if path:  # Make sure the path exists
                for segment_i in range(len(path) - 1):
                    pygame.draw.line(screen, COLORS[color], path[segment_i], path[segment_i+1], 3)
    
    # Event handle
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_x, mouse_y = event.pos
            
            # Color choose
            if 20 <= mouse_x <= 20 + COLOR_BUTTON_SIZE:
                for color_index, color in enumerate(COLORS_ORDER):
                    button_y = 50 + color_index * (COLOR_BUTTON_SIZE + 10)
                    if button_y <= mouse_y <= button_y + COLOR_BUTTON_SIZE:
                        if len(points[color]) < 2:
                            selected_color = color
                        break
            
            # Click on start button
            elif START_BUTTON.collidepoint(mouse_x, mouse_y) and not solution_computed:
                print("Start to compute the path")
                start_time = time.perf_counter()
                
                # Prioritize the user placed order
                print("=" * 40)
                user_order_result = user_order_connect()
                
                print("=" * 40)
                priority_result = priority_connect()
                
                print("=" * 40)
                greedy_result = greedy_connect()
                
                print("=" * 40)
                ga_result = run_genetic_algorithm()
                
                end_time = time.perf_counter()
                
                # Choose the most connected count result
                results = [
                    (len(user_order_result), user_order_result, "User placed order"),
                    (len(priority_result), priority_result, "Priority connect"),
                    (len(greedy_result), greedy_result, "Greedy algorithm"),
                    (len(ga_result), ga_result, "Genetic algorithm")
                ]

                # used connected count as reference to sort from the 4 algorithm
                results.sort(reverse=True, key=lambda x: x[0])
                
                # If user placement order solution is the same as the best solution, prioritize user placement order
                best_count = results[0][0]
                user_order_count = len(user_order_result)
                
                if user_order_count == best_count and user_order_result:
                    result = user_order_result
                    method = "User placement order"
                else:
                    best_count, result, method = results[0]
                
                if result:
                    solution.update(result)
                    solution_computed = True
                    print("=" * 40)
                    print(f"Choose {method} result: {len(result)} color")
                    print(f"Computation completed, total: {end_time - start_time:.2f} seconds")
                    print(f"The final connected colors: {list(result.keys())}")
                else:
                    print("All algorithm can't find valid connection")
            
            # Reset button
            elif RESET_BUTTON.collidepoint(mouse_x, mouse_y):
                for color in COLORS_ORDER:
                    points[color].clear()
                selected_color = None
                solution.clear()
                solution_computed = False
                color_placement_order.clear()
                print("Reset done")
            
            # Click on grid
            else:
                grid_x = round((mouse_x - GRID_START_X) / DOT_SPACING)
                grid_y = round((mouse_y - GRID_START_Y) / DOT_SPACING)
                
                # Click on the grid and no solution and selected a color
                if (0 <= grid_x < GRID_SIZE and 0 <= grid_y < GRID_SIZE and 
                    not solution_computed and selected_color):
                    
                    screen_pos = (GRID_START_X + grid_x * DOT_SPACING, 
                                  GRID_START_Y + grid_y * DOT_SPACING)
                    
                    # Check if this position is occupied
                    occupied = False
                    for color in COLORS_ORDER:
                        if screen_pos in points[color]:
                            occupied = True
                            break
                    
                    # If not occupied, place the color if still available
                    if not occupied:
                        # If it's the first point, added into the placement order
                        if len(points[selected_color]) == 0:
                            add_color_to_placement_order(selected_color)
                        
                        points[selected_color].append(screen_pos)
                        if len(points[selected_color]) == 2:
                            selected_color = None
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()