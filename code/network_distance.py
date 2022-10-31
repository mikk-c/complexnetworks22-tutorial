import copy, os, subprocess, itertools
import numpy as np
import pandas as pd
import networkx as nx
from pyemd import emd as _emd
from scipy import spatial
from scipy.sparse import csgraph
from collections import defaultdict
from multiprocessing import Pool, Manager

def _ge_Q(network):
   A = nx.adjacency_matrix(network).todense().astype(float)
   return np.linalg.pinv(csgraph.laplacian(np.matrix(A), normed = False))

def _ge_ml_collapse_vectors(vectors, network, couplings = set([]), coupling_style = "clique", layer_jump_weight = defaultdict(lambda : defaultdict(lambda : 1))):
   for n in network.nodes:
      network.nodes[n]["layer"] = list(dict(network[n]).values())[0]["layer"]
   for i in range(len(vectors)):
      vectors[i] = defaultdict(lambda: 0, vectors[i])
   if coupling_style == "clique":
      for coupling in couplings:
         for pair in itertools.combinations(coupling, 2):
            if layer_jump_weight[network.nodes[pair[0]]["layer"]][network.nodes[pair[1]]["layer"]] == np.inf:
               for i in range(len(vectors)):
                  vectors[i][pair[0]] += vectors[i][pair[1]]
   elif coupling_style == "chain":
      for coupling in couplings:
         for i in range(len(coupling) - 1):
            if layer_jump_weight[network.nodes[coupling[i]]["layer"]][network.nodes[coupling[i + 1]]["layer"]] == np.inf:
               for j in range(len(vectors)):
                  vectors[j][coupling[i]] += vectors[j][coupling[i + 1]]
   return [{k: v for k, v in d.items() if v} for d in vectors]

def _ge_ml_preprocess(network, couplings = set([]), coupling_style = "clique", layer_jump_weight = defaultdict(lambda : defaultdict(lambda : 1))):
   # Phase 1: Infer in which layer a node is
   for n in network.nodes:
      network.nodes[n]["layer"] = list(dict(network[n]).values())[0]["layer"]
   # Phase 2: Add the coupling edges as a special "coupling" layer
   # At the same time, if some layers have instantaneaous spread, contract their coupled nodes
   network = nx.MultiGraph(network)
   if coupling_style == "clique":
      for coupling in couplings:
         for pair in itertools.combinations(coupling, 2):
            if pair[0] in network.nodes and pair[1] in network.nodes: # if I can't find the node, it was already collapsed in the correct one
               if layer_jump_weight[network.nodes[pair[0]]["layer"]][network.nodes[pair[1]]["layer"]] == np.inf:
                  network = nx.contracted_nodes(network, pair[0], pair[1])
               else:
                  network.add_edge(pair[0], pair[1], layer = "coupling")
   elif coupling_style == "chain":
      for coupling in couplings:
         for i in range(len(coupling) - 1):
            if layer_jump_weight[network.nodes[coupling[i]]["layer"]][network.nodes[coupling[i + 1]]["layer"]] == np.inf:
               nx.contracted_nodes(network, coupling[i], coupling[i + 1], copy = False)
            else:
               network.add_edge(coupling[i], coupling[i + 1], layer = "coupling")
   if nx.number_connected_components(network) > 1:
      raise ValueError("Node vector distance is only valid if calculated on a network with a single connected component. The network passed has more than one.")
   # Phase 2b: As a result of the previous phase, network might be a multigraph. So we need to collapse the parallel edges according to the chosen collapse style
   _network = nx.Graph()
   for e in network.edges(data = True):
      if _network.has_edge(e[0], e[1]):
         _network[e[0]][e[1]]["layer"].add(e[2]["layer"])
      else:
         _network.add_edge(e[0], e[1], layer = set([e[2]["layer"],]))
   nx.set_node_attributes(_network, {n: {"layer": network.nodes[n]["layer"]} for n in _network.nodes})
   network = _network
   network.graph["coupled"] = True
   return network

def _ge_ml_Q(network, collapse_style = sum, layer_weight = defaultdict(lambda : 1), layer_jump_weight = defaultdict(lambda : defaultdict(lambda : 1))):
   # Phase 3: Build B, the oriented incidence matrix of the input network
   B = nx.incidence_matrix(network, oriented = True)
   # Phase 4: Build W, the edge weight diagonal matrix
   edge_weights = []
   for e in network.edges(data = True):
      if "coupling" in e[2]["layer"]:
         edge_weights.append(layer_jump_weight[network.nodes[e[0]]["layer"]][network.nodes[e[1]]["layer"]])
      else:
         edge_weights.append(collapse_style(layer_weight[l] for l in e[2]["layer"]))
   W = np.diag(edge_weights)
   # Phase 5: Create the weighted Laplacian
   L = B * W * B.T
   # Phase 6: Generate Q and pass it to normal GE
   return np.linalg.pinv(L)

# GE interprets edge weights as capacities: doubling all edge weights lowers GE by a factor of sqrt(2) in a chain graph
def ge(src, trg, network, Q = None, normed = True, multilayer = False, couplings = set([]), coupling_style = "clique", collapse_style = sum, layer_weight = defaultdict(lambda : 1), layer_jump_weight = defaultdict(lambda : defaultdict(lambda : 1))):
   if multilayer:
      if not "coupled" in network.graph:
         network = _ge_ml_preprocess(network, couplings, coupling_style, layer_jump_weight)
      if Q is None:
         Q = _ge_ml_Q(network, collapse_style, layer_weight, layer_jump_weight)
   if nx.number_connected_components(network) > 1:
      raise ValueError("Node vector distance is only valid if calculated on a network with a single connected component. The network passed has more than one.")
   src = np.array([src[n] if n in src else 0. for n in network.nodes()])
   trg = np.array([trg[n] if n in trg else 0. for n in network.nodes()])
   if normed:
      src = src / src.sum() if src.sum() > 0 else src
      trg = trg / trg.sum() if trg.sum() > 0 else trg
   diff = src - trg
   if Q is None:
      Q = _ge_Q(network)
   return np.sqrt(diff.T.dot(np.array(Q).dot(diff)))

def _gft_ml_v(network, collapse_style = max, layer_weight = defaultdict(lambda : 1), layer_jump_weight = defaultdict(lambda : defaultdict(lambda : 1))):
   # Phase 3: Build B, the oriented incidence matrix of the input network
   B = nx.incidence_matrix(network, oriented = True)
   # Phase 4: Build W, the edge weight diagonal matrix
   edge_weights = []
   for e in network.edges(data = True):
      if "coupling" in e[2]["layer"]:
         edge_weights.append(1 / layer_jump_weight[network.nodes[e[0]]["layer"]][network.nodes[e[1]]["layer"]])
      else:
         edge_weights.append(1 / collapse_style(layer_weight[l] for l in e[2]["layer"]))
   W = np.diag(edge_weights)
   # Phase 5: Create the weighted Laplacian
   L = B * W * B.T
   l, v = np.linalg.eig(L)
   idx = l.argsort()
   l = l[idx]
   v = np.diag(l).dot(v[:,idx].T)
   return v

def _gft_v(network):
   A = nx.adjacency_matrix(network).todense().astype(float)
   laplacian = csgraph.laplacian(np.matrix(A), normed = False)
   l, v = np.linalg.eig(laplacian)
   idx = l.argsort()
   l = l[idx]
   v = np.diag(l).dot(v[:,idx].T)
   return v

# GFT interprets edge weights as costs: doubling all edge weights increases GFT by a factor of 2 in a chain graph
def gft(src, trg, network, v = None, normed = True, linkage = "euclidean", multilayer = False, couplings = set([]), coupling_style = "clique", collapse_style = max, layer_weight = defaultdict(lambda : 1), layer_jump_weight = defaultdict(lambda : defaultdict(lambda : 1))):
   if multilayer:
      if not "coupled" in network.graph:
         network = _ge_ml_preprocess(network, couplings = couplings, coupling_style = coupling_style, layer_jump_weight = layer_jump_weight)
      if v is None:
         v = _gft_ml_v(network, collapse_style = collapse_style, layer_weight = layer_weight, layer_jump_weight = layer_jump_weight)
   if nx.number_connected_components(network) > 1:
      raise ValueError("Node vector distance is only valid if calculated on a network with a single connected component. The network passed has more than one.")
   src = np.array([src[n] if n in src else 0. for n in network.nodes()])
   trg = np.array([trg[n] if n in trg else 0. for n in network.nodes()])
   if normed:
      src = src / src.sum() if src.sum() > 0 else src
      trg = trg / trg.sum() if trg.sum() > 0 else trg
   if v is None:
      v = _gft_v(network)
   src = src.dot(v)
   trg = trg.dot(v)
   if linkage == "euclidean":
      return spatial.distance.euclidean(src, trg)
   elif linkage == "cosine":
      return spatial.distance.cosine(src, trg)
   elif linkage == "pearson":
      return spatial.distance.correlation(src, trg)
   else:
      raise ValueError("No valid linkage strategy. Possible values: euclidean (default), cosine, pearson.")

def _spl_ml_preprocess(network, couplings = set([]), coupling_style = "clique", collapse_style = max, layer_weight = defaultdict(lambda : 1), layer_jump_weight = defaultdict(lambda : defaultdict(lambda : 1))):
   network = _ge_ml_preprocess(network, couplings = couplings, coupling_style = coupling_style, layer_jump_weight = layer_jump_weight)
   for e in network.edges(data = True):
      if "coupling" in e[2]["layer"]:
         network[e[0]][e[1]]["weight"] = 1 / layer_jump_weight[network.nodes[e[0]]["layer"]][network.nodes[e[1]]["layer"]]
      else:
         network[e[0]][e[1]]["weight"] = 1 / collapse_style(layer_weight[l] for l in e[2]["layer"])
   return network

def _spl(x):
   x[2][x[1]] = dict(nx.shortest_path_length(x[0], source = x[1], weight = "weight"))

def calculate_spl(network, nonzero_nodes, n_proc, return_as_dict = False):
   manager = Manager()
   shortest_path_lengths = manager.dict()
   pool = Pool(processes = n_proc)
   _ = pool.map(_spl, [(network, n, shortest_path_lengths) for n in nonzero_nodes])
   pool.close()
   pool.join()
   shortest_path_lengths = dict(shortest_path_lengths)
   if return_as_dict:
      return {j: {i: shortest_path_lengths[j][i] for i in nonzero_nodes} for j in nonzero_nodes}
   else:
      return np.array([[shortest_path_lengths[i][j] if j in shortest_path_lengths[i] else np.inf for i in nonzero_nodes] for j in nonzero_nodes], dtype = "float64")

# The current pathfinding implementation ignores edge weights, thus they do not affect EMD nor SPL
def emd(src, trg, network, shortest_path_lengths = None, n_proc = 1, normed = True, multilayer = False, couplings = set([]), coupling_style = "clique", collapse_style = max, layer_weight = defaultdict(lambda : 1), layer_jump_weight = defaultdict(lambda : defaultdict(lambda : 1))):
   if multilayer and not "coupled" in network.graph:
      network = _spl_ml_preprocess(network, couplings = couplings, coupling_style = coupling_style, collapse_style = collapse_style, layer_weight = layer_weight, layer_jump_weight = layer_jump_weight)
   if nx.number_connected_components(network) > 1:
      raise ValueError("Node vector distance is only valid if calculated on a network with a single connected component. The network passed has more than one.")
   nonzero_nodes = list((set(src) | set(trg)) & set(network.nodes))
   src = np.array([src[n] if n in src else 0. for n in nonzero_nodes], dtype = float)
   trg = np.array([trg[n] if n in trg else 0. for n in nonzero_nodes], dtype = float)
   if normed:
      src = src / src.sum() if src.sum() > 0 else src
      trg = trg / trg.sum() if trg.sum() > 0 else trg
   if shortest_path_lengths is None:
      shortest_path_lengths = calculate_spl(network, nonzero_nodes, n_proc)
   else:
      shortest_path_lengths = np.array([[shortest_path_lengths[i][j] for i in nonzero_nodes] for j in nonzero_nodes], dtype = "float64")
   return _emd(src, trg, shortest_path_lengths)

def _standardize_node_vectors(src, trg, normed):
   if normed:
      src_std = {x: src[x] / sum(src.values()) for x in src}
      trg_std = {x: trg[x] / sum(trg.values()) for x in trg}
   else:
      src_sum = sum(src.values())
      trg_sum = sum(trg.values())
      if src_sum < trg_sum:
         src_std = {x: src[x] * (trg_sum / src_sum) for x in src}
         trg_std = copy.deepcopy(trg)
      else:
         src_std = copy.deepcopy(src)
         trg_std = {x: trg[x] * (src_sum / trg_sum) for x in trg}
   return src_std, trg_std

def _spl_mean(src, trg, shortest_path_lengths, normed):
   total_path_length = sum(src[origin] * trg[dest] * shortest_path_lengths[dest][origin] for origin in src for dest in trg)
   if normed:
      return total_path_length
   else:
      return total_path_length / sum(src.values())

def _make_pathlengths(spl, src, trg, delete_s = None, delete_t = None):
   pathlengths = defaultdict(set)
   if delete_s is not None:
      for length in spl:
         for path in spl[length]:
            if path[0] != delete_s:
               pathlengths[length].add(path)
   elif delete_t is not None:
      for length in spl:
         for path in spl[length]:
            if path[1] != delete_t:
               pathlengths[length].add(path)
   else:
      for s in src:
         for t in trg:
            pathlengths[spl[s][t]].add((s,t))
   return pathlengths

def _spl_greedy(src, trg, ascending, shortest_path_lengths):
   pathlengths = _make_pathlengths(shortest_path_lengths, src, trg)
   total_path_length = 0.0
   while len(pathlengths) > 0:
      if ascending:
         pathlength = min(pathlengths)
      else:
         pathlength = max(pathlengths)
      weights = {pair: min(src[pair[0]], trg[pair[1]]) for pair in pathlengths[pathlength]}
      row = max(weights, key = weights.get)
      total_path_length += (weights[row] * pathlength)
      src[row[0]] -= weights[row]
      trg[row[1]] -= weights[row]
      if src[row[0]] == 0:
         del src[row[0]]
         pathlengths = _make_pathlengths(pathlengths, None, None, delete_s = row[0])
      if trg[row[1]] == 0:
         del trg[row[1]]
         pathlengths = _make_pathlengths(pathlengths, None, None, delete_t = row[1])
   return total_path_length

def spl(src, trg, network, linkage = "avg", shortest_path_lengths = None, n_proc = 1, normed = True):
   if nx.number_connected_components(network) > 1:
      raise ValueError("Node vector distance is only valid if calculated on a network with a single connected component. The network passed has more than one.")
   nonzero_nodes = list((set(src) | set(trg)) & set(network.nodes))
   src_std, trg_std = _standardize_node_vectors(src, trg, normed)
   if shortest_path_lengths is None:
      shortest_path_lengths = calculate_spl(network, nonzero_nodes, n_proc, return_as_dict = True)
   if linkage == "single":
      return _spl_greedy(src_std, trg_std, True, shortest_path_lengths)
   elif linkage == "complete":
      return _spl_greedy(src_std, trg_std, False, shortest_path_lengths)
   elif linkage == "avg":
      return _spl_mean(src_std, trg_std, shortest_path_lengths, normed)
   else:
      raise ValueError("No valid linkage strategy. Possible values: avg (default), single, complete.")

def _make_mapp_runs(src, trg, shortest_path_lengths):
   runs = []
   while sum(src.values()) > 1e-12:
      spl_df = pd.DataFrame(shortest_path_lengths).unstack().reset_index()
      spl_df.columns = ("trg", "src", "spl")
      spl_df = spl_df[spl_df["src"].isin(src) & spl_df["trg"].isin(trg)]
      spl_df["allocable_weight"] = spl_df.apply(lambda x : min(src[x["src"]], trg[x["trg"]]), axis = 1)
      spl_df = spl_df[spl_df["allocable_weight"] > 0].sort_values(by = ["spl", "allocable_weight"], ascending = [True, False])
      robot_starts = {} # vertex, robotid
      robot_ends = {} # vertex, robotid
      robot_weights = {} # robotid, weight
      cur_robot_id = 1
      while spl_df.shape[0] > 0:
         robot_starts[int(spl_df.iloc[0]["src"])] = cur_robot_id
         robot_ends[int(spl_df.iloc[0]["trg"])] = cur_robot_id
         robot_weights[cur_robot_id] = spl_df.iloc[0]["allocable_weight"]
         cur_robot_id += 1
         src[spl_df.iloc[0]["src"]] -= spl_df.iloc[0]["allocable_weight"]
         trg[spl_df.iloc[0]["trg"]] -= spl_df.iloc[0]["allocable_weight"]
         spl_df = spl_df[(spl_df["src"] != spl_df.iloc[0]["src"]) & (spl_df["trg"] != spl_df.iloc[0]["trg"])]
      runs.append({"starts": robot_starts, "ends": robot_ends, "weights": robot_weights})
   return runs

def _mapp_cost(G, runs):
   total_path_length = 0
   for cur_run in range(len(runs)):
      with open("temp.cpf", 'w') as f:
         f.write("V =\n")
         for n in G.nodes:
            start_code = runs[cur_run]["starts"][n] if n in runs[cur_run]["starts"] else 0
            end_code = runs[cur_run]["ends"][n] if n in runs[cur_run]["ends"] else 0
            f.write("(%d:-1)[%s:%s:%s]\n" % (n, start_code, end_code, end_code))
         f.write("E =\n")
         for e in G.edges:
            f.write("{%d,%d} (-1)\n" % (e[0], e[1]))
      bash_command = "./insolver_reLOC --input-file=temp.cpf --output-file=temp.out"
      process = subprocess.Popen(bash_command.split(), stdout = subprocess.PIPE)
      output, error = process.communicate()
      if error is None and b"Total cost = 0" in output:
         continue
      with open("temp.out", 'r') as f:
         for line in f:
            try:
               fields = line.strip().split()
               robot = int(fields[0])
               total_path_length += runs[cur_run]["weights"][robot]
            except:
               continue
      os.remove("temp.cpf")
      os.remove("temp.out")
   return total_path_length

#TODO: Slow, but probably I can't do much about it...
def mapp(src, trg, network, shortest_path_lengths = None, n_proc = 1, normed = True):
   if nx.number_connected_components(network) > 1:
      raise ValueError("Node vector distance is only valid if calculated on a network with a single connected component. The network passed has more than one.")
   nodemap = {list(network.nodes)[i]: i for i in range(len(network.nodes))}
   G = nx.relabel_nodes(network, nodemap)
   src = {nodemap[x]: src[x] for x in src}
   trg = {nodemap[x]: trg[x] for x in trg}
   nonzero_nodes = list(set(src) | set(trg))
   src, trg = _standardize_node_vectors(src, trg, normed)
   if shortest_path_lengths is None:
      shortest_path_lengths = calculate_spl(G, nonzero_nodes, n_proc, return_as_dict = True)
   runs = _make_mapp_runs(src, trg, shortest_path_lengths)
   return _mapp_cost(G, runs)

def _mmc_P(network, time_steps):
   A = nx.to_numpy_array(network)
   np.fill_diagonal(A, 1)
   P = A / A.sum(axis = 0)
   return np.linalg.matrix_power(P, time_steps)

def _mmc_matrix(v, network, P, time_steps):
   if P is None:
      P = _mmc_P(network, time_steps)
   sigma_sum = v.dot(P.dot(1.0 - P)) ** 2
   return sigma_sum

# MMC interprets edge weights as costs: increasing all edge weights increases MMC with diminishing returns in a chain graph
def mmc(src, trg, network, time_steps = None, P = None, normed = True):
   if nx.number_connected_components(network) > 1:
      raise ValueError("Node vector distance is only valid if calculated on a network with a single connected component. The network passed has more than one.")
   src = np.array([src[n] if n in src else 0. for n in network.nodes()])
   trg = np.array([trg[n] if n in trg else 0. for n in network.nodes()])
   if normed:
      src = src / src.sum()
      trg = trg / trg.sum()
   diff = src - trg
   if time_steps == None:
      time_steps = nx.diameter(network)
   sigma_src = _mmc_matrix(src, network, P, time_steps)
   sigma_trg = _mmc_matrix(trg, network, P, time_steps)
   sigma = (sigma_src + sigma_trg) / 2
   Q = np.diag(1.0 / sigma)
   return np.sqrt(diff.dot(Q.dot(diff)))

def _annihilation_Q(A):
   P = A / A.sum(axis = 0)
   l, v = np.linalg.eig(P)
   idx = l.argsort()
   l = l[idx]
   v = v[:,idx]
   stationary = v[:,-1] / np.sum(v[:,-1])
   P_inf = np.tile(stationary, (1, P.shape[0]))
   return np.array((np.identity(P.shape[0]) - (P - P_inf)) ** -1)

# The current implementation of annihilation ignores edge weights, thus they do not affect it
def annihilation(src, trg, network, Q = None, normed = True):
   if nx.number_connected_components(network) > 1:
      raise ValueError("Node vector distance is only valid if calculated on a network with a single connected component. The network passed has more than one.")
   src = np.array([src[n] if n in src else 0. for n in network.nodes()])
   trg = np.array([trg[n] if n in trg else 0. for n in network.nodes()])
   if normed:
      src = src / src.sum()
      trg = trg / trg.sum()
   if Q is None:
      A = nx.adjacency_matrix(network).todense().astype(float)
      Q = _annihilation_Q(A)
   return _annihilation(src, trg, Q)

def _annihilation(src, trg, Q):
   diff = src - trg
   return np.real(np.sqrt(diff.T.dot(Q).dot(diff)))

def correlation(src, trg, network, shortest_path_lengths = None, n_proc = 1):
   src = np.array([src[n] if n in src else 0. for n in network.nodes], dtype = float)
   trg = np.array([trg[n] if n in trg else 0. for n in network.nodes], dtype = float)
   src_ = src - np.mean(src)
   trg_ = trg - np.mean(trg)
   if shortest_path_lengths is None:
      shortest_path_lengths = calculate_spl(network, network.nodes, n_proc)
   W = 1 / np.exp(shortest_path_lengths)
   numerator = (W * np.outer(src_, trg_)).sum()
   denominator_src = np.sqrt((W * np.outer(src_, src_)).sum())
   denominator_trg = np.sqrt((W * np.outer(trg_, trg_)).sum())
   return numerator / (denominator_src * denominator_trg)

def moran(v, network, shortest_path_lengths = None, n_proc = 1, kernel = "neighbors"):
   v = np.array([v[n] if n in v else 0. for n in network.nodes], dtype = float)
   return _moran(v, network, shortest_path_lengths = shortest_path_lengths, n_proc = n_proc, kernel = kernel)

def _moran(v, network, shortest_path_lengths = None, n_proc = 1, kernel = "neighbors"):
   v_ = v - np.mean(v)
   if kernel == "neighbors":
      W = np.array(nx.adjacency_matrix(network).todense())
   else:
      if shortest_path_lengths is None:
         shortest_path_lengths = calculate_spl(network, network.nodes, n_proc)
      if kernel == "linear":
         W = 1 / shortest_path_lengths
      elif kernel == "exponential":
         W = 1 / np.exp(shortest_path_lengths - 1)
      np.fill_diagonal(W, 0)
   W = W / W.sum(axis = 1)
   return (W * np.outer(v_, v_)).sum() / (v_ ** 2).sum()

def _resistance(network):
   L = csgraph.laplacian(np.matrix(nx.adjacency_matrix(network).todense().astype(float)), normed = False)
   Q = np.linalg.pinv(L)
   zeta = np.diag(Q)
   u = np.ones(zeta.shape[0])
   return np.array((np.matrix(u).T * zeta) + (np.matrix(zeta).T * u) - (2 * Q))

def variance(v, network, shortest_path_lengths = None, n_proc = 1, kernel = "geodesic"):
   v = np.array([v[n] if n in v else 0. for n in network.nodes], dtype = float)
   v = v / v.sum()
   if kernel == "geodesic":
      if shortest_path_lengths is None:
         shortest_path_lengths = calculate_spl(network, network.nodes, n_proc)
   elif kernel == "resistance":
      if shortest_path_lengths is None:
         shortest_path_lengths = _resistance(network)
   return (np.outer(v, v) * (shortest_path_lengths ** 2)).sum() / 2
