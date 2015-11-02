if ~exist('optgpu', 'var'),
    optgpu = 0;
end

% pcd (ML)
demo_mnist('mnist_orig',optgpu,'pcd',1000,0.01,0.005,0.3,0.05,0.0001,300,100,0.01,1,1,1,1,10,500);

% CD-percloss
demo_mnist('mnist_orig',optgpu,'cdpl',1000,0.01,0.005,0.1,0.1,0.0001,300,100,0.01,1,1,1,0,10);

% mrnn
demo_mnist('mnist_orig',optgpu,'mrnn',1000,0.01,0.005,0.1,0.1,0.0001,300,100,0.01,1,1,1,1,5,500,5,5,0.25,0.25,1);

% hybrid
demo_mnist('mnist_orig',optgpu,'hybrid',1000,0.01,0.005,0.1,0.1,0.0001,300,100,0.01,1,1,1,1,10,500,5,5,0.1,0.1,0.5);

