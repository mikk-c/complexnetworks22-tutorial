#=
G: The adjacency matrix of the graph, best if in sparse CSC format
o, v1, v2: numerical vectors, must have length equal to G's dimensions
=#

module NetworkDistance
   using Laplacians, LinearAlgebra, SparseArrays, Distances, Statistics, LightGraphs

   function ge(G, o, method)
      if method == "base"
         L = lap(G);
         Q = pinv(Matrix(L));
         return sqrt(dot(o', Q * o));
      elseif method == "approxchol"
         solver = approxchol_lap(G);
      elseif method == "augtree"
         solver = augTreeLap(G);
      elseif method == "kmp"
         solver = KMPLapSolver(G);
      elseif method == "cg"
         solver = cgLapSolver(G);
      end
      x = solver(o);
      return sqrt(dot(o', x));
   end

   function er_fast(G, method, approx = 0.3)
      if method == "approxchol"
         f = approxchol_lap(G);
      elseif method == "augtree"
         f = augTreeLap(G);
      elseif method == "kmp"
         f = KMPLapSolver(G);
      elseif method == "cg"
         f = cgLapSolver(G);
      end
      n = size(G, 1);
      k = ceil(Int, log(n) / (approx ^ 2));
      U = wtedEdgeVertexMat(G);
      m = size(U, 1);
      R = randn(Float64, m, k);
      UR = U' * R;
      V = zeros(n, k);
      for i in 1:k
         V[:,i] = f(UR[:,i]);
      end
      return pairwise(Euclidean(), V'; dims = 2) .^ 2 / k;
   end

   function er_exact(G)
      L = lap(G);
      Q = pinv(Matrix(L));
      zeta = diag(Q);
      u = ones(size(zeta, 1));
      return (u * zeta') + (zeta * u') - (2 * Q);
   end

   function variance(G, o, method)
      v = o / sum(o);
      if method == "exact"
         ER = er_exact(G);
      else
         ER = er_fast(G, method);
      end
      return sum((v .* v') .* (ER .^ 2)) / 2;
   end

   function network_correlation(G, v1, v2, method)
      v1_ = v1 .- mean(v1);
      v2_ = v2 .- mean(v2);
      if method == "spl"
         ER = LightGraphs.Parallel.dijkstra_shortest_paths(squash(Graph(G)), 1:size(G, 1)).dists;
      elseif method == "exact"
         ER = er_exact(G);
      else
         ER = er_fast(G, method);
      end
      W = 1 ./ exp.(ER);   
      numerator = sum(W .* (v1_ .* v2_'));
      denominator1 = sqrt(sum(W .* (v1_ .* v1_')));
      denominator2 = sqrt(sum(W .* (v2_ .* v2_')));
      return numerator / (denominator1 * denominator2);
   end

end
