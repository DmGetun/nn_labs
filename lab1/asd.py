def get_child(indexes, parents):
    child_1 = [20] * len(parents[0])
    child_2 = [20] * len(parents[0])
    
    start, stop = indexes[0], indexes[1]
    parent_1, parent_2 = parents[0], parents[1]
    
    child_1[start:stop] = parent_2[start:stop]
    child_2[start:stop] = parent_1[start:stop]

    insert_index = 0
    for i in range(len(parent_2)):
        vertex = parent_2[i]
        if (vertex not in child_2):
            child_2[insert_index] = vertex
            insert_index += 1
            if insert_index == start: insert_index = stop
            
    return child_1, child_2


parent1 = [8,4,7,3,6,2,5,1,9,0]
parent2 = [0,1,2,3,4,5,6,7,8,9]
indexes = (3, 8)

print(get_child(indexes, (parent1, parent2)))