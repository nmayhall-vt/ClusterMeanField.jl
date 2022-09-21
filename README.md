# ClusterMeanField

[![Build Status](https://github.com/nmayhall-vt/ClusterMeanField.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/nmayhall-vt/ClusterMeanField.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/nmayhall-vt/ClusterMeanField.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/nmayhall-vt/ClusterMeanField.jl)

### Installation
1. Download

	```bash
	git clone git@github.com:nmayhall-vt/ClusterMeanField.jl.git
	cd ClusterMeanField.jl/
	```


2. Create python virtual environment which will hold the PYSCF executable.   where `-tauto` let's Julia pick the max number of threads. Use `-t N` to select `N` manually. Removing defaults to 1 thread. 

	```bash
	virtualenv -p python3 venv
	source venv/bin/activate
	pip install -r requirements.txt
	julia --project=./ -tauto 
  	```


3. Build PyCall from Julia REPL

  	```julia
	using Pkg; Pkg.build("PyCall")
	```


4. Run tests
	```
	Pkg.test()
	```
