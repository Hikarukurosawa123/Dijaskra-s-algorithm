#include "main.h"

void example(Graph* gr)
{
    add(gr, "Spadina");

    add(gr, "Kipling");
    add(gr, "Yorkdale");
    add(gr,"Bay");
    
    add(gr,"Union");
    
    add(gr,"Bloor-Yonge");
    
    add(gr,"Kennedy");
    add(gr,"Sheppard-Yonge");
    add(gr,"Finch");
    add(gr,"Don Mills");



    

    update(gr, "Kipling", "Spadina", 12);

    update(gr, "Spadina", "Yorkdale", 8);
    update(gr, "Spadina", "Bay", 2);
    update(gr, "Spadina", "Union", 4);

    update(gr, "Bay", "Bloor-Yonge", 1);
    update(gr, "Union", "Bloor-Yonge", 3);
    update(gr, "Bloor-Yonge", "Kennedy", 14);
    update(gr, "Bloor-Yonge", "Sheppard-Yonge", 11);
    update(gr, "Sheppard-Yonge", "Finch", 3);
    update(gr, "Sheppard-Yonge", "Don Mills", 6);

   // print_gr(gr);

    
    char **r = plan_route(gr, "Kipling", "Don Mills");
    print_route(r); 
    /*
    print_route(r, "Kipling");

    //Bye bye Bay!
    disrupt(gr, "Bay");
    print_gr(gr);
    r = plan_route(gr, "Kipling", "Don Mills");
    print_route(r, "Kipling");

    //Hello new stations
    update(gr, "Yorkdale", "Sheppard West", 5);
    update(gr, "Sheppard West", "Sheppard-Yonge", 4);
    print_gr(gr);
    r = plan_route(gr, "Kipling", "Don Mills");
    print_route(r, "Kipling");
    */
   //free_gr(gr);
    
}

int main(){
    //Building the graph in Figure 1
    Graph *gr = initialize_graph();
    char input;
    printf("Input an integer that corresponds to the desired action: \n");
    printf("1. add station \n2. update station (add connection between the stations with speicified weightings \n3. delete a station \n4. print current stations and connections \n5. find route \n6. run example\n7. exit");
    //scanf("%c", &input);
            char* station = malloc(sizeof(char)*MAX_LEN);
            char* start = malloc(sizeof(char)*MAX_LEN);
            char* dest = malloc(sizeof(char)*MAX_LEN);
            int weight; 
            char **r;

    while(1)
    {
        scanf("%c", &input);

        switch(input)
    {
        case '1':
            printf("enter station name");
            scanf("%s", station);
            add(gr, station);
            printf("continue? Press Y or N");
            break;
        case '2':
        
            printf("enter starting station");
            scanf("%s", start);
            printf("enter destination station");
            scanf("%s", dest);
            printf("enter weighting");
            scanf("%d", &weight);
            update(gr, start, dest, weight);
            printf("continue? Press Y or N");

            
            break;
        case '3':
            printf("enter name of the deleting station");
            scanf("%s", station);
            disrupt(gr, station);
            printf("continue? Press Y or N");

            break;
        case '4':
            print_gr(gr);
            printf("continue? Press Y or N");

            break;
        case '5':
            printf("enter starting station");
            scanf("%s", start);
            printf("enter destination station");
            scanf("%s", dest);
            r = plan_route(gr, start, dest);
            print_route(r);
            printf("continue? Press Y or N");

            break;
        case '6':
            example(gr);
            printf("continue? Press Y or N");

            break;
        case '7'://exit
            break;
        case 'N':
            break;
        case 'Y':
            printf("1. add station \n2. update station (add connection between the stations with speicified weightings \n3. delete a station \n4. print current stations and connections \n5. find route \n6. run example\n7. exit");

            break;
        
        
    }

    if(input == 'N' || input == '7')
    {
        break;
    }

    }


    
        
    return 0;
}