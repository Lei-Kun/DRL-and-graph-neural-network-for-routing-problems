from gurobipy import *
#from create_tsp_instance import creat_instance
import numpy as np
def solve_euclidian_tsp(points, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan TSP problem to optimality using the MIP formulation
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate
    :return:
    """

    n = len(points)

    # Callback - use lazy constraints to eliminate sub-tours

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist((i, j) for i, j in model._vars.keys() if vals[i, j] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            if len(tour) < n:
                # add subtour elimination constraint for every pair of cities in tour
                model.cbLazy(quicksum(model._vars[i, j]
                                      for i, j in itertools.combinations(tour, 2))
                             <= len(tour) - 1)

    # Given a tuplelist of edges, find the shortest subtour

    def subtour(edges):
        unvisited = list(range(n))
        cycle = range(n + 1)  # initial length has 1 more city
        while unvisited:  # true if list is non-empty
            thiscycle = []
            neighbors = unvisited
            while neighbors:
                current = neighbors[0]
                thiscycle.append(current)
                unvisited.remove(current)
                neighbors = [j for i, j in edges.select(current, '*') if j in unvisited]
            if len(cycle) > len(thiscycle):
                cycle = thiscycle
        return cycle

    # Dictionary of Euclidean distance between each pair of points

    dist = {(i,j) :
        math.sqrt(sum((points[i][k]-points[j][k])**2 for k in range(2)))
        for i in range(n) for j in range(i)}

    m = Model()
    m.Params.outputFlag = False

    # Create variables

    vars = m.addVars(dist.keys(), obj=dist, vtype=GRB.BINARY, name='e')
    for i,j in vars.keys():
        vars[j,i] = vars[i,j] # edge in opposite direction

    m.addConstrs(vars.sum(i,'*') == 2 for i in range(n))


    m._vars = vars
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    m.optimize(subtourelim)

    vals = m.getAttr('x', vars)
    selected = tuplelist((i,j) for i,j in vals.keys() if vals[i,j] > 0.5)

    tour = subtour(selected)
    assert len(tour) == n

    return m.objVal, tour


def solve_all_gurobi(dataset):
    results = []
    tour=[]
    for i, instance in enumerate(dataset):
        print ("Solving instance {}".format(i))
        result,tur= solve_euclidian_tsp(instance)
        results.append(result)
        tour.append(tur)
    return results,tour

def random_tsp(n_nodes, random_seed=None):
    if random_seed is None:
        random_seed = np.random.randint(123456789)
    np.random.seed(random_seed)
    # torch.manual_seed(random_seed)
    # data = torch.FloatTensor(n_nodes,2).uniform_(0,1)
    data = np.random.uniform(0, 1, (n_nodes, 2))
    return data
datas = []
for i in range(10000):
    data =random_tsp(20)
    datas.append(data)

a = solve_all_gurobi(datas)
a = np.array(a).mean()
print(a)
