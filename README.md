# Project Name

This software project accompanies the research paper, [Probabilistic Transformers](https://arxiv.org/abs/2010.15583).

This project contains implementation of a Pytorch module for the probabilistic attention update proposed in the above paper. 

## Documentation

Runs an update of the probabilistic version of attention based on a Mixture of Gaussians model.  
It accepts the following parameters during a forward pass:  
* q: A tensor of queries with dims N, G, C, H  
* zeta: A tensor of keys (query/key Gaussian means) with dims N, G, C, H  
* alpha: A scalar (see special case above) or tensor of query/key Gaussian precisions with dims N, G, C, H  
* mu: A tensor of value Gaussian means with dims N, G, Cv, H  
* beta: A scalar (see special case above) or tensor of value Gaussian precisions with dims N, G, C, H  
* pi: A tensor of mixture component priors with dims N, G, H, H  
* v_init: A tensor of initial vals for the values with dims N, G, Cv, H (optional)  
* v_fixed: A tensor of fixed vals for the values with dims N, G, Cv, H (optional)  
* zeta_prior_precision: A tensor of precisions for the Gaussian prior over zeta with dims N, G, C, H (optional)  
* mu_prior_precision: A tensor of precisions for the Gaussian prior over mu with dims N, G, Cv, H (optional)  
* q_pos_emb: A tensor of query positional embeddings with dims C, H, H  
* zeta_pos_emb: A tensor of key positional embeddings with dims C, H, H  
* v_pos_emb: A tensor of value positional embeddings with dims Cv, H, H  
* nonzero_wts_mask: A boolean indexing tensor for setting weight matrix values to zero (where mask value is false) with dims H, H  

And returns the following output tensor:   
* Updated values with dims N, G, Cv, H if no position embedding (v_pos_emb=None) else N, G, 2*Cv, H  

Notably, this layer is equivalent to a standard dot product attention (without position embeddings) when:  
* uniform_query_precision = True  
* uniform_value_precision = True  
* magnitude_priors = True  
* alpha = 1/sqrt(C) (Could be a scalar to save some memory)  
* beta = 0 (Could be a scalar to save some memory)  
* v_init = None  
* v_fixed = None  

## Getting Started 

The module is in the file probabilisticattention.py.
It can be imported as any other Pytorch layer.
