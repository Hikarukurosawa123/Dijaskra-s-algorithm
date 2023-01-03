import math 
import pygame 

class Particle: #class for the circles
    def __init__(self, x, y, text, curr_cost):
        self.x = x
        self.y = y

        self.colour = (0, 0, 255)
        self.thickness = 1
        self.text = text
        self.curr_cost = str(curr_cost)
        self.text_surface = base_font.render(self.text, True, BLACK)
        self.cost_surface = base_font.render(self.curr_cost, True, BLACK)


        self.size = max(50, self.text_surface.get_width())



    def display(self,screen):
        pygame.draw.circle(screen, self.colour, (self.x, self.y), self.size, self.thickness)

class Line: #class for the lines connecting the station nodes 
    def __init__(self, name, destination, source_x, source_y, dest_x, dest_y, text, source_size, dest_size):
        self.name = name 
        self.destination = destination 
        self.source_x = source_x
        self.source_y = source_y
        self.dest_x = dest_x
        self.dest_y = dest_y
        self.text = str(text)
        self.text_surface = base_font.render(self.text, True, BLACK)
        self.thickness = 1
        self.color = BLACK
        self.source_size = source_size
        self.dest_size = dest_size
        
    def display(self,screen):
        pygame.draw.line(screen, self.color, (self.source_x, self.source_y), (self.dest_x, self.dest_y))


class graph: #stores the vertexes (stations) into a list
    def __init__(self):
        self.adjacent = []
        self.count = 0


    def add_vertex(self, name, x, y, objects):
        new_vertex = vnode(name, x, y, objects)
        self.adjacent.append(new_vertex)
        self.count = self.count + 1
        
        return new_vertex 

    def initialize_cost(self, start, objects):
        for station in self.adjacent:
            if station.name == start: 
                station.curr_cost = 0
            else:
                station.curr_cost = str(math.inf)

        for station in objects:
            if station.text == start:
                station.cost_surface = base_font.render(str(0), True, BLACK)
            else:
                station.cost_surface = base_font.render(str(math.inf), True, BLACK)

    def update(self, cost, source, destination):
        for station in self.adjacent:
            if station.name == source:
                station.add_edge(destination, cost)

class queue: #the edge nodes are added into the priority list and the path with the smallest cost is brought to the front using the heapify function 
    def __init__(self):
        self.priority = [] 

class enode: #represents the weighted connection between the stations. 
    def __init__(self, source, destination, cost):
        self.source = source
        self.destination = destination
        self.cost = cost
        self.dest_node_cost = str(math.inf)
        self.line = None


class vnode: #represents the stations
    def __init__(self, name, x, y, objects):
        self.name = name 
        self.curr_cost = str(math.inf)
        self.visited = 0 
        self.edges = []
        self.x = x
        self.y = y
        self.circle = Particle(self.x, self.y, name, self.curr_cost)
        objects.append(self.circle)
        self.shortest_path = None
        
    def update_cost(self, cost, shortest_path):
        if self.curr_cost != str(math.inf) and cost != str(math.inf):
            if int(self.curr_cost)  > int(cost):
                self.curr_cost  = str(cost)
                self.shortest_path = shortest_path
        else:
            self.curr_cost  = str(cost)
            self.shortest_path = shortest_path



    def add_edge(self, Graph, destination, cost, objects, lines):
        new_edge = enode(self.name, destination, cost)
        self.edges.append(new_edge)
        index_source = find_index(Graph, self.name)
        index_dest = find_index(Graph, destination)
        for station in objects: 
            if station.text == Graph.adjacent[index_source].name:
                source_x = Graph.adjacent[index_source].x + station.size
                source_y = Graph.adjacent[index_source].y
                source_size = station.size

            elif station.text == Graph.adjacent[index_dest].name:
                dest_x = Graph.adjacent[index_dest].x - station.size
                dest_y = Graph.adjacent[index_dest].y
                dest_size = station.size

        line = Line(self.name, destination, source_x, source_y, dest_x, dest_y, cost, source_size, dest_size)
        new_edge.line = line
        lines.append(line)
        




def print_graph(Graph):
    for name in Graph.adjacent:
        print(name.name)

def print_edges(Graph):
    for station in Graph.adjacent:
        print(station.name, station.edges)

def find_index(Graph, name):
    for i, station in enumerate(Graph.adjacent):
        if station.name == name:
            return i 
    return False
def add_queue(Graph, start, queue):
    for station in Graph.adjacent: 
        if station.name == start:
            if station.visited == 0: 
                for edges in station.edges:
                    queue.priority.append(edges)
    return queue

def heapify(Queue):
    min_cost = math.inf
    index = None 
    for i,station in enumerate(Queue.priority):
        if station.dest_node_cost < min_cost:
            min_cost = station.dest_node_cost
            index = i

    
    min = Queue.priority.pop(index)

    Queue.priority.insert(0, min)

    return Queue

def find_shortest(Graph, start, end, Queue, objects):
    Graph.initialize_cost(start, objects)
    index = find_index(Graph, start) #finds the beginnign of the trip 
    curr_station = start
    Queue = add_queue(Graph, Graph.adjacent[index].name, Queue)
    path = []
    vertexes = []
    priority = None

    while curr_station != end and len(Queue.priority) != 0: #terminate once foudn the destination station or the priroirty queue length becomes zero 
        #visit the destination node and update the cost using the current node cost (if already visited) + edge cost 
        for enode in Queue.priority:
            index_dest = find_index(Graph, enode.destination) #finds the beginnign of the trip 
            index_source = find_index(Graph, enode.source)
            enode.dest_node_cost = int(enode.cost) + int(Graph.adjacent[index_source].curr_cost)
            Graph.adjacent[index_dest].update_cost(int(enode.cost) + int(Graph.adjacent[index_source].curr_cost), enode.source)
 
        Queue = heapify(Queue)
        priority = Queue.priority.pop(0)
        index = find_index(Graph, priority.destination)
        vertexes.append(Graph.adjacent[index].curr_cost)
        path.append(priority)

        curr_station = priority.destination
           
        Queue = add_queue(Graph, curr_station, Queue)
    shortest = back(Graph,end)
    index = find_index(Graph, "k")

    return objects, vertexes, shortest, path 


def back(Graph, end):
    station = end
    index = find_index(Graph, station)
    route = [station]
    while station is not None:
        route.append(Graph.adjacent[index].shortest_path)
        index = find_index(Graph, Graph.adjacent[index].shortest_path)
        station = Graph.adjacent[index].shortest_path
    return route #return the shortest route 
                
            




def example(objects, Graph, Queue, lines):
    Graph.add_vertex("a", 50, 300, objects)
    Graph.add_vertex("b", 150, 200, objects)
    Graph.add_vertex("c", 300, 200, objects)
    Graph.add_vertex("d", 450, 200, objects)
    Graph.add_vertex("e", 600, 200, objects)
    Graph.add_vertex("f", 750, 200, objects)
    Graph.add_vertex("g", 900, 200, objects)
    Graph.add_vertex("h", 1050, 200, objects)
    Graph.add_vertex("i", 1200, 200, objects)
    Graph.add_vertex("j", 1350, 300, objects)
    Graph.add_vertex("k", 600, 550, objects)

    Graph.adjacent[0].add_edge(Graph,"j", 28, objects, lines)

    Graph.adjacent[0].add_edge(Graph,"k", 14, objects, lines)

    Graph.adjacent[0].add_edge(Graph,"b", 1, objects, lines)
    Graph.adjacent[1].add_edge(Graph,"c", 2, objects, lines)
    Graph.adjacent[2].add_edge(Graph,"d", 3, objects, lines)
    Graph.adjacent[3].add_edge(Graph,"e", 5, objects, lines)
    Graph.adjacent[4].add_edge(Graph,"f", 5, objects, lines)
    Graph.adjacent[5].add_edge(Graph,"g", 5, objects, lines)
    Graph.adjacent[6].add_edge(Graph,"h", 5, objects, lines)
    Graph.adjacent[7].add_edge(Graph,"i", 5, objects, lines)
    Graph.adjacent[8].add_edge(Graph,"j", 5, objects, lines)
    Graph.adjacent[3].add_edge(Graph,"k", 5, objects, lines)


    Graph.adjacent[10].add_edge(Graph,"j", 15, objects, lines)
    objects, vertexes, shortest, route = find_shortest(Graph, "a", "j", Queue, objects)
    return objects, vertexes, shortest, route 



if __name__ == "__main__":
    pygame.init()
    SCREEN_WIDTH = 1500
    SCREEN_HEIGHT = 600
    Graph = graph()
    Queue = queue()
    
    base_font = pygame.font.Font(None, 32)
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    running = True
    objects = []
    lines = []
    user_text = ""
    allow_type = False
    show = False
    text_surface = None
    BLACK = (0,0,0)
    RED = (255, 0, 0)
    objects, vertexes, shortest, route = example(objects, Graph, Queue, lines)    

    length_shortest = len(shortest)
    


    while running:
        time = pygame.time.get_ticks()
        screen.fill((255,255,255))
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_presses = pygame.mouse.get_pressed()
                if mouse_presses[0]:
                    x,y = pygame.mouse.get_pos()
                    circle = Particle(x, y)
                    objects.append(circle)                    
                    
            if event.type == pygame.KEYDOWN:
                # Check for backspace
                if event.key == pygame.K_BACKSPACE:
                    # get text input from 0 to -1 i.e. end.
                    user_text = user_text[:-1]
                    
                elif event.key == pygame.K_RETURN:
                    text_surface = base_font.render(user_text, True, BLACK)
                    circle = objects[-1]
                    circle.text = text_surface 
                    user_text = ""
                    
                    
                else: 
                    user_text += event.unicode
                          
            
        for circle in objects:
            circle.display(screen)     
            screen.blit(circle.text_surface, circle.text_surface.get_rect(center = (circle.x, circle.y)))
            #use the path dest_node_cost to update the ccost of the vertex nodes
            screen.blit(circle.cost_surface, circle.cost_surface.get_rect(center = (circle.x, circle.y-20)))

        for line in lines:
            line.display(screen)    
            
            screen.blit(line.text_surface, ((line.dest_x - (line.dest_x - line.source_x )/2), (line.dest_y - (line.dest_y - line.source_y)/2) - 23))

        pygame.display.flip()
        #get the value of the dest_node_cost (dest_node_cost information available in the edgenode)
        pygame.time.wait(2000)

        for path in route: 
            for line in lines: 
                if path.line == line: 
                    line.color = RED
                    line.display(screen)
                    for circle in objects: 
                        if path.line.destination == circle.text:
                            rect = pygame.Rect(circle.x - circle.cost_surface.get_width()/2, circle.y-20 - circle.cost_surface.get_height()/2, circle.cost_surface.get_width() + 4, circle.cost_surface.get_height())
                            pygame.draw.rect(screen, (255,255,255), rect, 0)
                            if(circle.curr_cost != str(math.inf) and path.dest_node_cost != str(math.inf)):
                                if int(circle.curr_cost) > int(path.dest_node_cost):
                                    circle.curr_cost = path.dest_node_cost
                                    circle.cost_surface = base_font.render(str(path.dest_node_cost), True, BLACK) 
                                    screen.blit(circle.cost_surface, circle.cost_surface.get_rect(center = (circle.x, circle.y-23)))
                                else: 
                                    circle.cost_surface = base_font.render(str(circle.curr_cost), True, BLACK) 
                                    screen.blit(circle.cost_surface, circle.cost_surface.get_rect(center = (circle.x, circle.y-23)))

                            else:                                 
                                circle.curr_cost = path.dest_node_cost
                                circle.cost_surface = base_font.render(str(path.dest_node_cost), True, BLACK) 
                                screen.blit(circle.cost_surface, circle.cost_surface.get_rect(center = (circle.x, circle.y-23)))
                            
                    pygame.display.flip()

                    pygame.time.wait(2000)

        for i in range(length_shortest-1):  #station is a str variable 
            for line in lines: 
                if shortest[length_shortest-1-i] == line.name and line.destination == shortest[length_shortest-i-2]: 
                    line.color = (0,255,0)
                    line.display(screen)
                
        pygame.display.flip()

        pygame.time.wait(10000)
        running = False








