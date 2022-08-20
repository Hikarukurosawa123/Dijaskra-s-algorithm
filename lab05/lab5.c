#include "lab5.h"
typedef struct Queue{
    struct Enode *head; //keeps track of the head 
}Queue;
//this is a test run 
Queue* initialize_queue()
{
    Queue* q = calloc(1, sizeof(Queue));

    q->head = calloc(1,sizeof(Enode)); 
    q->head->curr = calloc(2,sizeof(char)); 
    q->head->next = NULL; 
 
    return q;
}

Graph* add_cost(Graph *gr, char* start, int* start_index)
{
    for (int i = 0; i < gr->count; i++)//initalize the distance for each node and add edges into the priority queue
    {
        if(strcmp(gr->adj_list[i]->station, start)==0)
        {
            gr->adj_list[i]->cost = 0; //set as infinity for all other nodes
            *start_index  = i;

        }
        else
        {
            gr->adj_list[i]->cost = 5001; //set as infinity for all other nodes

        }
    }
    return gr;

}

void add_queue(Queue* queue, Enode* edge_node, Vnode* vertex_node)
{
    Enode* prev;
    if(queue->head == NULL)//first node in the queue
    {
        queue->head = calloc(1, sizeof(Enode));
        queue->head->weight = edge_node->weight+vertex_node->cost;//adds the value of the weight and the cost  
        strcpy(queue->head->vertex, edge_node->vertex);
        queue->head->curr = calloc(1,sizeof(char));
        queue->head->next = NULL;

        
    }
    else
    {//not workign from this point on 
        Enode* head = queue->head;
        while(head!= NULL)//compare with the node 
        {
            if(edge_node->weight+vertex_node->cost < head->weight)//compare costs of next move and place the smaller value in front
            {
                if(prev!=NULL)//if it's not added to the front 
                {
                    Enode* temp = calloc(1, sizeof(Enode)); //this comes in front

                    temp->weight = edge_node->weight+ vertex_node->cost;
                    strcpy(temp->vertex, edge_node->vertex);
                    temp->curr = calloc(strlen(vertex_node->station)+1,sizeof(char));
                    strcpy(temp->curr, vertex_node->station);

                    prev->next = temp; 
                    temp->next = head;

                    
                    break;

                }
                
                Enode* temp = calloc(1, sizeof(Enode)); //this comes in front
                temp->weight = edge_node->weight+ vertex_node->cost;
                strcpy(temp->vertex, edge_node->vertex);
                //strcpy(temp->curr, vertex_node->station);//keeps track of the origin 
                temp->curr = calloc(strlen(vertex_node->station)+1,sizeof(char));
                strcpy(temp->curr , vertex_node->station);//keeps track of the origin 

                temp->next = head;//set the node in front of the node with higher weighting 
                //no need to change the head because it is already at head
                queue->head = temp;
                break;

            }
            else if(head->next!=NULL)
            {
                prev = head; 


                head = head->next;
            }
            else//if the weighting is the highest 
            {
                Enode* temp = calloc(1,sizeof(Enode));

                temp->weight = edge_node->weight+ vertex_node->cost;

                strcpy(temp->vertex, edge_node->vertex);
                temp->curr = calloc(strlen(vertex_node->station)+1,sizeof(char));

                strcpy(temp->curr , vertex_node->station);//keeps track of the origin 


                head->next = temp; //add temp as the last node 
                break;

            }
            
        }
    }
}
void find_start_index(Graph* gr, char*start, int* start_index)
{
    
    for(int i= 0; i<gr->count; i++)
    {
        
        if(strcmp(gr->adj_list[i]->station, start)==0)
        {
      
            *start_index=i;
            
        }
        
    }
    
    
}
void add_all_queue(Graph* gr, Queue* queue, Vnode* vertex_node)//adds all the edges inside the vertex_node
{
    Enode* head = vertex_node->edges;
    Enode* queue_head = queue->head;
    int x = -1; 
    int* start = &x; 

    while(queue_head != NULL)//update vertex_node cost by assigning the queue->head cost to vertex_node cost if the destination of the edge node happens to be the vertex_node (receives the accumulated cost upuntil that point)
    {         
        if(strcmp(queue_head->vertex, vertex_node->station)==0)//if the vertex node station is the same as the destination
        {//update vertex_node cost
        //when proceeding 
                        //printf("this is entered");

            
            if(vertex_node->cost > queue_head->weight)//if the updated cost is larger than the path cost from a different edge node
            {
                vertex_node->cost = queue_head->weight; 
                find_start_index(gr, queue_head->curr, start);       
      
                Vnode* temp = calloc(1, sizeof(Vnode));
                if(*start!=-1)//if not at start
                {
                   strcpy(temp->station,gr->adj_list[*start]->station);
                    vertex_node->prev = temp;

                }                
                else
                {
                    temp = NULL;
                    vertex_node->prev = temp;

                }
            }
        }
        queue_head = queue_head->next;
    }

    if(vertex_node->visited!=1)//if the edges nodes of the vertex_node hasn't been added yet
    {
    

        while(head!= NULL)
        {
            add_queue(queue, head, vertex_node);
            head = head->next;
        }
          
        

    }

    vertex_node->visited = 1;

    


}

void print_queue(Queue* q)
{
    Queue* queue = q;
      while(queue->head!= NULL)
   {
       printf("%d", queue->head->weight);
       queue->head = queue->head->next;
   }

}

char **plan_route(Graph *gr, char *start, char *dest){
    /*
    steps:
    1. find the starting node
    2. prev is null for starting node 
    3. set the cost value for all the vertex nodes equal to 5001 (infinity)
    4. Add the next moves to the priority queue in order of the costs
    */
    char** route = calloc(1, sizeof(char*)); 
   int x = 0;
   int* start_index = &x;
   int s = 0; 
   int* end_index = &s;
   gr = add_cost(gr, start, start_index); //gets start_index and assigns the cost for each vertex 
   Queue* queue = initialize_queue();


    add_all_queue(gr, queue, gr->adj_list[*start_index]);//add all the edge nodes in the starting vertex node 
    gr->adj_list[*start_index]->prev = NULL;
    while(queue->head!=NULL)
    {
        if(strcmp(queue->head->vertex, dest)==0)//if the destination is found, break out of the loop 
        {
            Vnode* temp = calloc(1, sizeof(Vnode));
            strcpy(temp->station, queue->head->curr);
            find_start_index(gr,queue->head->vertex, end_index);
            gr->adj_list[*end_index]->prev = temp;
            break;
        }
        find_start_index(gr,queue->head->vertex, start_index); //gets the index of the destination node 

        add_all_queue(gr, queue, gr->adj_list[*start_index]);//add all the edge nodes associated with the vertex node
        queue->head = queue->head->next;
    }

    if(queue->head == NULL)
    {
        printf("no path found\n");
        return NULL;
    }
    else if(strcmp(queue->head->vertex, dest)==0)
    {
        printf("path found\n");
    }
   route[0] = gr->adj_list[*end_index]->station;
        Vnode* prev = gr->adj_list[*end_index];
        int count = 1;

    while(prev->prev!=NULL)
    {
        route = realloc(route, sizeof(char*)*(count+1));
        route[count] = prev->station;
        count++; 
        find_start_index(gr,prev->prev->station, start_index);

        prev = gr->adj_list[*start_index];
    }
    route = realloc(route, sizeof(char*)*(count+1));
    find_start_index(gr,start, start_index); //gets the index of the destination node 
    route[count] = gr->adj_list[*start_index]->station; 

    char** final = calloc(count+1, sizeof(char*));
    int o= 0; 
    for (int i = count; i != 0; i--)
    {
        final[o] = route[i];
        o++;
    }

    return final; 
    


}
void print_cost(Graph* gr)
{
    for(int i = 0; i < gr->count; i++)
    {
        printf("%d", gr->adj_list[i]->cost);
    }

    
}
int check(Graph *gr, char* station)
{
    for(int i = 0; i < gr->count; i++)
    {
        if(strcmp(gr->adj_list[i]->station, station)==0 )
        {
            return 1; // return true 
        }
    }
    return 0;
}

void add(Graph *gr, char *station){
 if(gr->adj_list== NULL) // if the adj_list is null 
    {
        //gr->adj_list = (Vnode**) malloc(sizeof(Vnode*));//create new adjacent list 
        Vnode* adj_list = (Vnode* )calloc(1, sizeof(Vnode));
        strcpy(adj_list->station, station);
        adj_list->visited = 0;
        gr->adj_list = calloc(1, sizeof(Vnode));
        gr->adj_list[0] = adj_list;
        gr->adj_list[0]->edges = NULL;


        //printf("%s", gr->adj_list[0]->station);

        gr->count ++; 
        return;
    }
    else if(check(gr, station)!=1)//if the station already exists
    {
        gr->count++; 
        Vnode* adj_list = calloc(1, sizeof(Vnode));//create new adjacent list 
        strcpy(adj_list->station, station);//add on the count index 
         adj_list->visited = 0;

        gr->adj_list = realloc(gr->adj_list, sizeof(Vnode)*gr->count);
        gr->adj_list[gr->count-1] = adj_list;
        gr->adj_list[gr->count-1]->edges = NULL;
    }

}



void update(Graph *gr, char *start, char *dest, int weight){
    //things to accomplish 
    //1. iterate through all the nodes in the adjancy list 
    //2. for each node, iterate through the edges, and update the value of the weight if the start and destination exists
    //3. if the edge is null or if the destination does not already exist, add a new edge node
    for (int i = 0; i<gr->count; i++)//itertate through each node
    {
        if(strcmp(gr->adj_list[i]->station, start)==0)//if the starting vertex node exists
        {
            if(gr->adj_list[i]->edges == NULL)//if an edge does not already exist 
            {
                if(weight!= 0)//if the weight is not zero, add the edge node 
                {
                    Enode* edge_node = calloc(1, sizeof(Enode));
                    edge_node->weight = weight;//changes the value of the weight
                    strcpy(edge_node->vertex, dest);
                    edge_node->next = NULL;
                    gr->adj_list[i]->edges = edge_node;//go to next edge
                    //printf("%s", gr->adj_list[i]->edges->vertex);

                }
            }
            else
            {
                Enode* head = gr->adj_list[i]->edges;//keep track of the head
                while(gr->adj_list[i]->edges!=NULL)
                {
                    if (strcmp(gr->adj_list[i]->edges->vertex, dest)==0)//if the edge to the destination already exists   
                    {
                        if(weight == 0)
                        {
                            Enode* temp = gr->adj_list[i]->edges;
                            gr->adj_list[i]->edges = gr->adj_list[i]->edges ->next;//go to next edge
                            free(temp);//delete the edge node 

                        }
                        else//if the weight is not zero, update the weight value
                        {
                            gr->adj_list[i]->edges->weight = weight;//changes the value of the weight
                            gr->adj_list[i]->edges = gr->adj_list[i]->edges ->next;//go to next edge

                        }

                    }
                    else//if the start is the same but the destination is different 
                    {//need to keep track of the head node
                        while(gr->adj_list[i]->edges->next!=NULL)//should make this code more neat afterwards 
                        {
                            if (strcmp(gr->adj_list[i]->edges->vertex, dest)==0)//if the edge to the destination already exists   
                                {
                                    if(weight == 0)
                                    {
                                        Enode* temp = gr->adj_list[i]->edges;
                                        gr->adj_list[i]->edges = gr->adj_list[i]->edges ->next;//go to next edge
                                        free(temp);//delete the edge node 

                                    }
                                    else//if the weight is not zero, update the weight value
                                    {
                                        gr->adj_list[i]->edges->weight = weight;//changes the value of the weight
                                        gr->adj_list[i]->edges = gr->adj_list[i]->edges ->next;//go to next edge

                                    }

                                }
                            else
                            {
                                gr->adj_list[i]->edges = gr->adj_list[i]->edges->next;//proceed until the end, but what if the edge already exists?
                            }
                        }
                        //add new edgenode to the end 
                        Enode* edge_node = calloc(1, sizeof(Enode));
                        edge_node->weight = weight;
                        strcpy(edge_node->vertex, dest);
                        edge_node->next = NULL;
                        gr->adj_list[i]->edges->next = edge_node;//go to next edge
                        gr->adj_list[i]->edges = gr->adj_list[i]->edges->next;

                    }
                }
                gr->adj_list[i]->edges = head;//set the edge to the beginning

            }
    

        }        
        
    }
}

void disrupt(Graph *gr, char *station){
    //need to remove the edge in both directions(need to iterate through all the adjacency lists)
    for(int i = 0; i<gr->count; i++)
    {   
        if(strcmp(gr->adj_list[i]->station, station)==0)//if the station is the vertex to be removed
        {
            while(gr->adj_list[i]->edges!=NULL)
            {
                Enode* temp = gr->adj_list[i]->edges;
                gr->adj_list[i]->edges = gr->adj_list[i]->edges->next;
                free(temp);
                temp = NULL;
            }
            Vnode* vertex = gr->adj_list[i];
            //reassign the nodes
            for(int x = i+1; x< gr->count; x++)
            {
                gr->adj_list[i] = gr->adj_list[x];
            }
            gr->count--;
            gr->adj_list = realloc(gr->adj_list, sizeof(Vnode*)*gr->count);

            free(vertex);
            vertex = NULL;
        }
        //create two cases: one if the connectino is at the beginingn and one in between or at the end? 
    
    }

    for(int i = 0; i < gr->count; i++)
    {    

        if(gr->adj_list[i]->edges!=NULL)//check the edge node from the other direction
        {
        
            if(strcmp(gr->adj_list[i]->edges->vertex, station)==0)//if its the first connection
            {
                
                Enode* temp = gr->adj_list[i]->edges;
                gr->adj_list[i]->edges = gr->adj_list[i]->edges->next;
                free(temp);
            }
            
            else
            {       
                Enode* head = gr->adj_list[i]->edges;
                while(gr->adj_list[i]->edges->next!=NULL)
                {
                    //need to check if the edge for the next node is null or not to strcmp 
                    if(strcmp(gr->adj_list[i]->edges->next->vertex, station)==0)
                    {
                        Enode* temp = gr->adj_list[i]->edges->next;
                        if(gr->adj_list[i]->edges->next->next != NULL)
                        {
                            gr->adj_list[i]->edges->next= gr->adj_list[i]->edges->next->next;
                            gr->adj_list[i]->edges = gr->adj_list[i]->edges->next;
                        }
                        else
                        {
                            gr->adj_list[i]->edges->next= NULL;
                            gr->adj_list[i]->edges = gr->adj_list[i]->edges->next;
                        }

                        free(temp);
                    }
                    else
                    {
                        gr->adj_list[i]->edges = gr->adj_list[i]->edges->next;


                    }
                }
                gr->adj_list[i]->edges = head;

        
            }
            
            
        
        }
        
        
}
}
