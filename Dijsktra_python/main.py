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
            #this assumes that the source is on the left and the destinatonl is at the right 
            if station.text == Graph.adjacent[index_source].name:
                if Graph.adjacent[index_source].x <= Graph.adjacent[index_dest].x:
                    source_x = Graph.adjacent[index_source].x + station.size
                else: 
                    source_x = Graph.adjacent[index_source].x - station.size

                source_y = Graph.adjacent[index_source].y
                source_size = station.size

            elif station.text == Graph.adjacent[index_dest].name:
                if Graph.adjacent[index_source].x <= Graph.adjacent[index_dest].x: #if dest on the right 
                    dest_x = Graph.adjacent[index_dest].x - station.size
                else: 
                    dest_x = Graph.adjacent[index_dest].x + station.size

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
        print("dest, end", priority.destination,end)

        curr_station = priority.destination
           
        Queue = add_queue(Graph, curr_station, Queue)
    shortest = back(Graph,end)
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
    objects = []
    Graph.adjacent = []
    Queue.priority = []

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
    
    BLACK = (0,0,0)
    RED = (255, 0, 0)
    #objects, vertexes, shortest, route = example(objects, Graph, Queue, lines)    

    reset = True
    while running:
        if reset: 
            proceed = True
            curr_station = None
            add_edge = False
            source_station = None
            dest_station = None
            answer = False
            enter = False
            adding = True
            cost = None
            ask_cost = False
            run_example = False #run the example if the button is pressed 
            reset = False
            objects = []
            vertexes = []
            shortest = []
            route = []
            lines = []
            user_text = ""
            allow_type = False
            show = False
            text_surface = None
            button_text = ""
            instruction = None
        time = pygame.time.get_ticks()
        screen.fill((255,255,255))
        rect_button = pygame.Rect(30, 30, 150, 60)
        pygame.draw.rect(screen, (0,0,0), rect_button, 1)
        button_text = base_font.render("find route", True, BLACK) 
        screen.blit(button_text, button_text.get_rect(center = rect_button.center))
        rect_example = pygame.Rect(200, 30, 150, 60)
        pygame.draw.rect(screen, (0,0,0), rect_example, 1)
        button_text = base_font.render("example", True, BLACK) 
        screen.blit(button_text, button_text.get_rect(center = rect_example.center))
        rect_reset = pygame.Rect(370, 30, 150, 60)
        pygame.draw.rect(screen, (0,0,0), rect_reset, 1)
        button_text = base_font.render("reset", True, BLACK) 
        screen.blit(button_text, button_text.get_rect(center = rect_reset.center))
        add = True # boolean for adding station nodes
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_presses = pygame.mouse.get_pressed()
                if mouse_presses[0]:
                    x,y = pygame.mouse.get_pos()
                    if x > rect_button.left and x < rect_button.right and y > rect_button.top and y < rect_button.bottom:
                        enter = True #find route 
                    elif x > rect_example.left and x < rect_example.right and y > rect_example.top and y < rect_example.bottom:
                        run_example = True
                    elif x > rect_reset.left and x < rect_reset.right and y > rect_reset.top and y < rect_reset.bottom:
                        reset = True
                        break
                    elif proceed: #if the mouse is pressed and the stations names for all the other nodes are filled 
                        for circle in objects:
                            if abs(circle.x - x) <= circle.size and abs (circle.y - y) <= circle.size: #if the mouse is pressed inside the circle 
                                add = False #need to identify which station is clicked by iterating thorugh the circles and mathching the name 
                                if source_station == None: 
                                    source_station = circle.text
                                    break
                                else: 
                                    
                                    dest_station = circle.text
                                    ask_cost = True
                                    break           
                            elif abs(circle.x - x) <= 2* circle.size and abs (circle.y - y) <=  2* circle.size: 
                                add = False
                                break
                                
                        if add:    
                            proceed = False
                            curr_station = Graph.add_vertex("", x, y, objects)

                    




                    
            if event.type == pygame.KEYDOWN:
                # Check for backspace
                if not enter: #before finding the routes 
                    if event.key == pygame.K_BACKSPACE:
                        # get text input from 0 to -1 i.e. end.
                        user_text = user_text[:-1]
                    elif event.key == pygame.K_RETURN:
                        if ask_cost:
                            cost = user_text
                            user_text = ""
                            index = find_index(Graph, source_station)
                            Graph.adjacent[index].add_edge(Graph, dest_station, cost, objects, lines)
                            dest_station = None
                            source_station = None
                            ask_cost = False
                            break
                        curr_station.name = user_text
                        text_surface = base_font.render(user_text, True, BLACK)
                        curr_station.circle.text_surface = text_surface 
                        curr_station.circle.text = user_text
                        user_text = ""
                        objects.append(curr_station.circle)
                        proceed = True    
                        
                    else: 
                        user_text += event.unicode
                else: #waiting for user input for the source and destination nodes 
                    if event.key == pygame.K_RETURN: #finding the route 
                        
                        if source_station == None:
                            source_station = user_text
                            user_text = ""
                        elif dest_station == None: 
                            dest_station = user_text
                            user_text = ""
                            answer = True
                            objects, vertexes, shortest, route = find_shortest(Graph, source_station, dest_station, Queue, objects)

                            enter = False
                            user_text = ""
                            source_station = False
                            dest_station = False

                    else: 
                        user_text += event.unicode
        if enter: 
            pass 
        elif not ask_cost:
            button_text = base_font.render("1. Click screen to create a station 2. Type station name and hit enter 3. Click two stations to create connection", True, RED) 
            instruction = pygame.Rect(20, 100, button_text.get_width()+10, 60)
            pygame.draw.rect(screen, (0,0,0), instruction, 1)
            screen.blit(button_text, button_text.get_rect(center = instruction.center))
        else: 
            button_text = base_font.render("Type the cost and hit enter", True, RED) 
            instruction = pygame.Rect(540, 30, button_text.get_width()+10, 60)
            pygame.draw.rect(screen, (0,0,0), instruction, 1)
            screen.blit(button_text, button_text.get_rect(center = instruction.center))

        if enter: 
            if source_station == None:
                button_text = base_font.render("Type Source Station and Hit Enter", True, RED) 
                instruction = pygame.Rect(540, 30, button_text.get_width()+10, 60)
                pygame.draw.rect(screen, (0,0,0), instruction, 1)
                screen.blit(button_text, button_text.get_rect(center = instruction.center))
            elif dest_station == None: 
                erase = pygame.Rect(540, 30, button_text.get_width()+10, 60)
                pygame.draw.rect(screen, (255,255,255), erase, 0)
                button_text = base_font.render("Type Dest Station and Hit Enter", True, RED) 
                instruction = pygame.Rect(540, 30, button_text.get_width()+10, 60)
                pygame.draw.rect(screen, (0,0,0), instruction, 1)
                screen.blit(button_text, button_text.get_rect(center = instruction.center))
            else: 
                erase = pygame.Rect(540, 30, button_text.get_width()+10, 60)
                pygame.draw.rect(screen, (255,255,255), erase, 0)


        if run_example:
            objects, vertexes, shortest, route = example(objects, Graph, Queue, lines)    
            answer = True
        
        for line in lines: #draws the black connectino lines
            line.display(screen)    
            screen.blit(line.text_surface, ((line.dest_x - (line.dest_x - line.source_x )/2), (line.dest_y - (line.dest_y - line.source_y)/2) - 23))

        for circle in objects:  #draws the station nodes
            circle.display(screen)     
            screen.blit(circle.text_surface, circle.text_surface.get_rect(center = (circle.x, circle.y)))
            #use the path dest_node_cost to update the ccost of the vertex nodes
            screen.blit(circle.cost_surface, circle.cost_surface.get_rect(center = (circle.x, circle.y-20)))


        #button for finding the route 
   
        pygame.display.flip()


        if answer: 
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
        
            for i in range(len(shortest)-1, 0, -1):  #station is a str variable 
                for line in lines: 
                    if shortest[i] == line.name and line.destination == shortest[i-1]: 
                        line.color = (0,255,0)
                        line.display(screen)
            answer = False
            
            pygame.display.flip()

            pygame.time.wait(5000)
            running = False




        #pygame.display.flip()
        #get the value of the dest_node_cost (dest_node_cost information available in the edgenode)










