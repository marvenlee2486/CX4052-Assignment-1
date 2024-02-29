import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description = "Add parameter")
parser.add_argument("-i", "--MAXITER", help = "Set the MaxIteration parameter (Default 100000)")
parser.add_argument("-s", "--SEED", help = "Set the Seed parameter (Default No)")
parser.add_argument("-f", "--FAIR", help = "Set to random generate alphas and betas (default is standard AIMD). Set the maximum value for alpha")
parser.add_argument("-u", "--USER", help = "Set the number of users")
parser.add_argument("-c", "--CAPACITY", help = "Set the capacity")
parser.add_argument("-o", "--OUTPUT", help = "Set the prefix name for output file")
parser.add_argument("-m", "--MODE", help = "0 means user will no enter or join at any moment, 1 means user will enter or leave at any moment")
args = parser.parse_args()

MAXITERATION = 100000
CAPACITY = 100
USER = 2
FAIR = 0
OUTPUT = "Test"
DYNAMIC_MODE = 0
if args.MAXITER:
    MAXITERATION = int(args.MAXITER)
if args.SEED:
    np.random.seed(int(args.SEED))
if args.USER:
    USER = int(args.USER)
if args.CAPACITY:
    CAPACITY = int(args.CAPACITY)
if args.FAIR:
    FAIR = float(args.FAIR)
if args.OUTPUT:
    OUTPUT = args.OUTPUT
if args.MODE:
    DYNAMIC_MODE = int(args.MODE)

def alpha_function(x, alpha, is_using):
    ## increasing function with faster rate
    # return alpha * np.power(x + 1, 2)
    # return alpha * np.power(x + 1,1.01)
    # return alpha * np.exp(x)

    ## increasing function with slower rate
    # return alpha * np.log2(x + 2)
    # return alpha * np.sqrt(x + 1)
    
    ## Constant constant rate
    # return alpha

    ## Decreasing function
    a = (np.log(0.01) - np.log(alpha)) / (CAPACITY / is_using.sum())
    return alpha * np.exp(a * np.power(x, 1))
    return alpha * np.reciprocal(x + 1)

def beta_function(w, beta, is_using):
    ## Constant 
    # return beta

    ## Modified Function
    a = (np.log(0.1) - np.log(1 - beta)) / (CAPACITY / is_using.sum()) # If dynamic mode is not on, is_using.sum() = USER
    # print(a)
    return 1 - (1 - beta) * np.exp(a * np.power(w,1))

def run_experiment(alphas, betas):
    '''
        Run the simulation
        
        Parameters:
            alphas: ndarray (n,1)
                The alpha for the n users
            betas:    ndarray (n,1)
                The beta for the n users

        Returns:    
            history: ndarray (MAXITERATION,n)
                The allocation vector of each iteration
            is_md: ndarray (MAXITERATION, 1)
                0 or 1 value with 0 indicates AI and 1 indicates MD
    '''
    if alphas.shape != betas.shape:
        raise ValueError("Alphas and Betas should have the same size")
    n = alphas.shape[0]
    history = np.zeros(shape = (n,MAXITERATION)) # stores the history of window size
    is_md = np.zeros(shape = MAXITERATION) # stores whether at that iteration is AI (0) or MD (1)
    x = np.random.rand(n) * CAPACITY / n # The window size of user
    w_k = np.zeros(shape = USER) # Store window size after previous MD

    # State whether user is using the network
    is_using = np.ones(shape = USER)
    probability = np.zeros(shape = USER)
    if DYNAMIC_MODE == 1:
        is_using = np.zeros(shape = USER)

    for iteration in range(MAXITERATION):
        if DYNAMIC_MODE == 1:
            probability += 0.001 / MAXITERATION
            roulette = np.random.rand(USER)
            for idx in np.where( (roulette <= probability) is True):
                probability[idx] = 0
                is_using[idx] = 1 - is_using[idx]

        x *= is_using

        if x.sum() > CAPACITY:
            betas2 = np.array([beta_function(w, b, is_using) for w, b in zip(w_k, betas)])
            x *= betas2
            w_k = x.copy()        
            is_md[iteration] = 1
            # print("MD")
        else:
            alphas2 = np.array([alpha_function(x, a, is_using) for x, a in zip(x, alphas)])
            x += alphas2
            # print("AI")
        # print("Iteration", iteration, ":", closestness(theoritical_fairness, usage))
        history[:,iteration] = x
    return history, is_md

def perron_frobenius(alphas, betas):
    if alphas.shape != betas.shape:
        raise ValueError("Alphas and Betas should have the same size")
    ev = np.divide(alphas, 1 - betas)
    # normalized = ev / np.linalg.norm(ev)
    return ev

def calculate_closestness(theoritical, experimental):
    v1 = theoritical / np.linalg.norm(theoritical)
    v2 = experimental / np.linalg.norm(experimental)
    return np.linalg.norm(v1 - v2)

def calculate_throughtput(history, is_md):
    throughput = 0
    for iteration in range(MAXITERATION):
        if is_md[iteration] == 1:
            continue
        allocation = history[:,iteration]
        if allocation.sum() <= CAPACITY:
            throughput += allocation.sum()

    print("Throughput : ", throughput)
    print("Maximium Throughput : ", (MAXITERATION - 1) * CAPACITY)
    print("# of AI:", MAXITERATION - is_md.sum(), ", # of MD:", is_md.sum())
    print("Percentage of Time Experiences MD:", is_md.sum() * 100 / MAXITERATION, "%")

# Graph Plotting
def plot_2_user_graph(data, fairness, is_multiplicative, title = None):
    fairness = fairness * CAPACITY / np.sqrt(np.sum(np.square(fairness)))
    plt.clf()
    plt.title(title)
    plt.scatter(data[0,0], data[1,0], 5, c = "black")
    plt.plot(data[0,:] , data[1,:])
    plt.plot([CAPACITY,0],[0,CAPACITY], label = "Bottleneck")
    plt.plot([0,CAPACITY],[0,CAPACITY], "--", label = "Most Fairness line")
    plt.plot([0,fairness[0]],[0,fairness[1]], label = "Theorectical Convergence")
    plt.xlabel("Allocation of User 1")
    plt.ylabel("Allocation of User 2")
    # idx = np.where(is_multiplicative == 1)[0][0] - 1
    plt.legend()
    # plt.xlim(0,CAPACITY)
    plt.savefig("diagram/" + OUTPUT + "_allocation_graph.png")
    
def plot_individual_graph(data, id):
    user = data[id,:]
    plt.clf()
    plt.plot(user)
    plt.xlabel("Iteration")
    plt.ylabel("Allocation of User " + str(id + 1))
    plt.savefig("diagram/" + OUTPUT + "_User" + str(id + 1) + ".png")

def plot_closeness_fairness_graph(theorectical, history):
    closeness_value = []
    for iter in range(MAXITERATION):
        closeness_value.append(calculate_closestness(theorectical, history[:,iter]))
    plt.clf()
    plt.plot(closeness_value)
    plt.xlabel("Iteration")
    plt.ylabel("Euclidean Distance")
    plt.title("Fairness graph")
    plt.savefig("diagram/" + OUTPUT + "_Fairness.png")

def main():
    betas = np.ones(shape = USER) * 0.5
    alphas = np.ones(shape = USER)
    if FAIR != 0:    
        betas = np.random.rand( USER) 
        alphas = np.random.rand( USER) * FAIR
    
    betas = np.array([0.3,0.5])
    alphas = np.array([2, 1])
    history, is_md = run_experiment(alphas, betas)
    fairness = perron_frobenius(alphas, betas)

    calculate_throughtput(history, is_md)

    plot_closeness_fairness_graph(fairness, history)
    if USER == 2:
        print("Plotting Allocation Graph")
        plot_2_user_graph(history,
                    fairness,
                    is_md,
                    "Parameter a1 = " +
                        str(round(alphas[0],2)) +
                        " , a2 = " +
                        str(round(alphas[1],2)) +
                        " , b1 = " +
                        str(round(betas[0],2)) +
                        " , b2 = " +
                        str(round(betas[1],2))
                    )
        plot_individual_graph(history, 0)
        plot_individual_graph(history, 1)
    # else:
        # for i in range(USER):
        #     plot_individual_graph(history, i)

if __name__ == "__main__":
    main()
