import numpy as np

# dataset 4

def load_d4():
    n = 200
    x_d4 = 3 * (np.random.rand(n, 4) - 0.5)
    y_d4 = (2 * x_d4[:, 0] - 1 * x_d4[:,1] + 0.5 + 0.5 * np.random.randn(n)) > 0
    y_d4 = 2 * y_d4 -1
    return x_d4, y_d4

def load_d5():
    n = 200
    x_d5 = 3 * (np.random.rand(n, 4) - 0.5)
    W = np.array([[ 2,  -1, 0.5,],
                  [-3,   2,   1,],
                  [ 1,   2,   3]])
    y_d5 = np.argmax(np.dot(np.hstack([x_d5[:,:2], np.ones((n, 1))]), W.T)
                            + 0.5 * np.random.randn(n, 3), axis=1)
    return x_d5, y_d5

def load_toy1(n):
    x = 3 * (np.random.rand(n, 2) - 0.5);
    radius = (x[:, 0])**2 + (x[:, 1])**2
    y = ((radius > 0.7 + 0.1 * np.random.randn(1, n)) & (radius < 2.2 + 0.1 * np.random.randn(1, n)))[0]
    y = 2 * y -1;
    return x, y

def load_toy6():
    d = 200;
    n = 180;
    # we consider 5 groups where each group has 40 attributes
    g = cell(5, 1);
    for i in range(1,length(g)):
        g[i-1] = (i-1)*40+1:i*40
    x = np.randn(n, d)
    noise = 0.5;
    # we consider feature in group 1 and group 2 is activated.
    w = [20 * randn(80, 1);
        zeros(120, 1);
        5 * rand];
    x_tilde = [x, ones(n, 1)];
    y = x_tilde * w + noise * randn(n, 1);
    lambda = 1.0;
    wridge = (x_tilde’*x_tilde + lambda * eye(d+1))\(x_tilde’ * y);
    cvx_begin
    variable west(d+1,1)
    minimize( 0.5 / n * (x_tilde * west - y)’ * (x_tilde * west - y) + ...
        lambda * (norm(west(g{1}), 2.0) + ...
        norm(west(g{2}), 2.0) + ...
        norm(west(g{3}), 2.0) + ...
        norm(west(g{4}), 2.0) + ...
        norm(west(g{5}), 2.0) ))
    cvx_end
    x_test = randn(n, d);
    x_test_tilde = [x_test, ones(n, 1)];
    y_test = x_test_tilde * w + noise * randn(n, 1);
    y_pred = x_test_tilde * west;
    mean((y_pred - y_test) .ˆ2)
    figure(1);
    clf;
    plot(west(1:d), ’r-o’)
    hold on
    plot(w, ’b-*’);
    plot(wridge, ’g-+’);
    legend(’group lasso’, ’ground truth’, ’ridge regression’)
    figure(2);
    clf;
    plot(y_test, y_pred, ’bs’);
    xlabel(’ground truth’)
    ylabel(’prediction’)
    fprintf(’carinality of w hat: %d\n’, length(find(abs(west) < 0.01)))
    fprintf(’carinality of w ground truth: %d\n’, length(find(abs(w) < 0.01)))

def sgd_loss(lam,w,x,y):
    loss = 0
    grad = 0
    for xi,yi in zip(x,y):#データサイズごと
        exp = np.exp(-xi.dot(w)*yi)
        loss += np.log(1+exp)
        grad += -exp*yi*xi/(1+exp)
    loss /= x.shape[0]
    grad /= x.shape[0]
    loss += lam * w.dot(w)
    grad += 2*lam*w
    return [loss, grad]

def sgd_mult_loss(lam,w,x,y):
    loss = 0
    grad = np.zeros((3,x.shape[1]))
    z = np.zeros((x.shape[0],3))
    for i in range(x.shape[0]):
        for j in range(3):
            z[i][j] = w[j].dot(x[i])
    softmax = np.exp(z)
    for softmaxi in softmax:
        softmaxi /= sum(softmaxi)
    for j in range(3):
        for xi,yi,softmaxi in zip(x,y,softmax):
            if j == yi:
                loss += -np.log(softmaxi[j])
                grad[j] += (softmaxi[j]-1)*xi
            else:
                grad[j] += softmaxi[j]*xi
    loss /= x.shape[0]
    grad /= x.shape[0]
    loss += lam * lam * np.linalg.norm(w)**2
    grad += 2*lam*w
    return [loss, grad]

def create_kernel(x,xz,a):
    n = x.shape[0]
    kernel = np.zeros(n)
    for i in range(n):
        kernel[i] += np.exp(-a*np.linalg.norm(x[i]-xz)**2)
    return kernel

def create_kernel_table(x,a):
    n = x.shape[0]
    table = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            table[i][j] += np.exp(-a*np.linalg.norm(x[i]-x[j])**2)
    return table

def create_kernel_matrix(table,z):
    n = z.shape[0]
    matrix = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            print(z[j])
            matrix[i][j] += table[i][z[j]]
    return matrix

# proximal gradient
def st_ops(mu, q):
  x_proj = np.zeros(mu.shape)
  for i in range(len(mu)):
    if mu[i] > q:
      x_proj[i] = mu[i] - q
    else:
      if np.abs(mu[i]) < q:
        x_proj[i] = 0
      else:
        x_proj[i] = mu[i] + q;
  return x_proj
