import torch 
import pandas as pd 
import itertools

def als_naive(df,rank:int =10,lamb:int =5):

    @torch.no_grad()
    @torch.jit.script
    def rmse_loss(X:torch.Tensor,Y: torch.Tensor,R:torch.Tensor,mask:torch.Tensor)-> float:
        P=X.T@Y
        E=(P-R)[mask]
        rmse= torch.sqrt(torch.mean(E**2))
        return float(rmse.detach().cpu())
    

    
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using: {device}')
    if device.type=='cuda':
        torch.cuda.empty_cache()        
        torch.cuda.reset_peak_memory_stats()

    piv = df.pivot(index='user_id',columns='item_id',values='rating')
    piv.index=piv.index.astype(int)
    piv.columns= piv.columns.astype(int)
    R=piv.sort_index(axis=0).sort_index(axis=1).to_numpy()

    R=torch.tensor(R,dtype=torch.float32,device=device)
    mask=~torch.isnan(R)
    n=piv.shape[0]
    m=piv.shape[1]
    X=torch.randn((rank,n),dtype=torch.float32,device=device)
    Y=torch.randn((rank,m),dtype=torch.float32,device=device)

    lam_eye=torch.eye(rank,dtype=torch.float32,device=device)*lamb

    item_mask=[torch.where(mask[j][0]) for j in range(n)]
    user_mask=[torch.where(mask[:,i]) for i in range(m)]

    i=0
    prev_loss=float('inf')
    while True:
            with torch.no_grad():
                for u in range(n):
                    rating_maks=mask[u,:]
                    Y_masked=Y[:,item_mask[u]]
                    X[:,u]=torch.linalg.solve(
                            torch.mm(Y_masked,Y_masked.T)+lam_eye,
                            torch.mv(Y_masked,R[u,rating_maks])
                    )
                for j in range(m):
                    rating_maks=mask[:,j]
                    X_masked=X[:,user_mask[j]]
                    Y[:,j]=torch.linalg.solve(
                        torch.mm(X_masked,X_masked.T)+lam_eye,
                        torch.mv(X_masked,R[rating_maks,j])
                    )
                rmse=rmse_loss(X,Y,R,mask)
                print(f'iter {i}: RMSE={rmse}')
                i+=1
                if abs(rmse-prev_loss)<1e-3:
                    break
                prev_loss=rmse
    return X,Y
    
@torch.no_grad()
@torch.jit.script
def update_users_batched(X: torch.Tensor,Y: torch.Tensor,u_idx: torch.Tensor,
                         i_idx: torch.Tensor,r_vals: torch.Tensor,lam_eye: torch.Tensor)->torch.Tensor:
    S=torch.einsum('am,bm->mab',Y,Y)

    rank=X.size(0)
    n=X.size(1)

    A=torch.zeros((n,rank,rank),device=X.device)


    A.index_add_(dim=0,index=u_idx,source=S.index_select(0,i_idx))

    A+=lam_eye.unsqueeze(0)

    b=torch.zeros((rank,n),dtype=X.dtype,device=X.device)
    b.index_add_(dim=1,index=u_idx,source=Y.index_select(1,i_idx)*r_vals.unsqueeze(0))
                 
    L=torch.linalg.cholesky(A)
    X_T=torch.cholesky_solve(b.T.unsqueeze(-1),L).squeeze(-1)
    return X_T.T

@torch.no_grad()
@torch.jit.script
def update_items_batched(X: torch.Tensor,Y: torch.Tensor,u_idx: torch.Tensor,
                         i_idx: torch.Tensor,r_vals: torch.Tensor,lam_eye: torch.Tensor)->torch.Tensor:
    S=torch.einsum('am,bm->mab',X,X)

    rank=Y.size(0)
    m=Y.size(1)

    A=torch.zeros((m,rank,rank),device=Y.device)
    A.index_add_(dim=0,index=i_idx,source=S.index_select(0,u_idx))

    A+=lam_eye.unsqueeze(0)

    b=torch.zeros((rank,m),dtype=Y.dtype,device=Y.device)
    b.index_add_(dim=1,index=i_idx,source=X.index_select(1,u_idx)*r_vals.unsqueeze(0))
                 
    L=torch.linalg.cholesky(A)
    Y_T=torch.cholesky_solve(b.T.unsqueeze(-1),L).squeeze(-1)
    return Y_T.T

@torch.no_grad()
@torch.jit.script
def rmse_loss(X:torch.Tensor,Y: torch.Tensor,u_idx: torch.Tensor,i_idx: 
              torch.Tensor,r_vals: torch.Tensor)-> float:
    pred= (X[:,u_idx]*Y[:,i_idx]).sum(dim=0)
    rmse= torch.sqrt(torch.mean((pred-r_vals)**2))
    return float(rmse.detach().cpu())



def als_paralel(df_users,df_items,df_ratings,n_users_total, m_items_total,rank:int =10,lamb:int =5, max_iter:int=100):

    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        torch.cuda.empty_cache()        
        torch.cuda.reset_peak_memory_stats()

    X = torch.randn(rank, n_users_total, device=device, dtype=torch.float32)
    Y = torch.randn(rank, m_items_total, device=device, dtype=torch.float32)

    u_idx=torch.tensor(df_users.values,dtype=torch.long,device=device)
    i_idx=torch.tensor(df_items.values,dtype=torch.long,device=device)
    r_vals=torch.tensor(df_ratings.values,dtype=torch.int64,device=device)

    lam_eye=torch.eye(rank,device=device)*lamb
    i=0
    prev_loss=float('inf')

    X.requires_grad_(False)
    Y.requires_grad_(False)
    u_idx.requires_grad_(False)
    i_idx.requires_grad_(False)
    r_vals.requires_grad_(False)
    lam_eye.requires_grad_(False)

    while True:
   

            X=update_users_batched(X,Y,u_idx,i_idx,r_vals,lam_eye)
            Y=update_items_batched(X,Y,u_idx,i_idx,r_vals,lam_eye)
            rmse=rmse_loss(X,Y,u_idx,i_idx,r_vals)
            print(f'iter {i}: RMSE={rmse}')
            i+=1
            if abs(rmse-prev_loss)<1e-3 or i>=max_iter:
                break
            prev_loss=rmse
    return X,Y

def evaluate_als(X: torch.Tensor, Y: torch.Tensor, test_df:pd.DataFrame):
   
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type=='cuda':
        torch.cuda.empty_cache()        
        torch.cuda.reset_peak_memory_stats()
    
    u_idx = torch.tensor(test_df['user_id'].values, dtype=torch.long, device=device)
    i_idx = torch.tensor(test_df['item_id'].values, dtype=torch.long, device=device)
    r_true = torch.tensor(test_df['rating'].values, dtype=torch.float32, device=device)

    pred_ratings = (X[:, u_idx].T * Y[:, i_idx].T).sum(dim=1)
    
    pred_ratings_clipped = pred_ratings.clamp(min=1.0, max=5.0).round()
    
    rmse = torch.sqrt(torch.mean((pred_ratings_clipped - r_true) ** 2)).item()
    
    return rmse, pred_ratings_clipped



def tune_als(train_df, val_df, rank_list, lambda_list, max_iter=20):
    device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using: {device}')
    if device.type=='cuda':
        torch.cuda.empty_cache()        
        torch.cuda.reset_peak_memory_stats()

    best_rmse = float('inf')
    best_rank = None
    best_lambda = None

    for rank, lamb in itertools.product(rank_list, lambda_list):
        print(f"Testing rank={rank}, lambda={lamb}...")

        n_users_total = int(max(train_df['user_id'].max(), val_df['user_id'].max()) + 1)
        n_items_total = int(max(train_df['item_id'].max(), val_df['item_id'].max()) + 1)
        X = torch.randn(rank, n_users_total, device=device)
        Y = torch.randn(rank, n_items_total, device=device)

        u_idx = torch.tensor(train_df['user_id'].values, dtype=torch.long, device=device)
        i_idx = torch.tensor(train_df['item_id'].values, dtype=torch.long, device=device)
        r_vals = torch.tensor(train_df['rating'].values, dtype=torch.float32, device=device)

        lam_eye = lamb * torch.eye(rank, device=device)
        train_rmse=float('inf')
        for _ in range(max_iter):
            X = update_users_batched(X, Y, u_idx, i_idx, r_vals, lam_eye)
            Y = update_items_batched(X, Y, u_idx, i_idx, r_vals, lam_eye)
            train_rmse = rmse_loss(X, Y, u_idx, i_idx, r_vals)
        

        test_rmse,_ = evaluate_als(X, Y, val_df)
        print(f" -> Train RMSE={train_rmse:.4f}")
        print(f" -> RMSE={test_rmse:.4f}")

        if test_rmse < best_rmse:
            best_rmse = test_rmse
            best_rank = rank
            best_lambda = lamb

    print(f"Best rank={best_rank}, best lambda={best_lambda}, RMSE={best_rmse:.4f}")
    return best_rank, best_lambda, best_rmse
