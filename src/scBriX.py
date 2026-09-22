import os
import math
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import numpy as np
from torch.nn import functional as F
from .decoder import Decoder
from .encoder import Encoder
from .sampler import sample_diffusion
from .util import MultiModalDataset, DynamicHybridLoss, UnpairedMultiModalDataset, to_adata
import scanpy as sc
from sklearn.decomposition import TruncatedSVD
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)


class NoiseScheduler():
    def __init__(self,
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear", device='cuda'):
        self.num_timesteps = num_timesteps

        if beta_schedule == "linear":
            self.betas = torch.linspace(
                beta_start, beta_end, num_timesteps + 1, dtype=torch.float32)
        elif beta_schedule == "quadratic":
            self.betas = torch.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps + 1, dtype=torch.float32) ** 2
        elif beta_schedule == 'cosine':
            self.betas = torch.from_numpy(betas_for_alpha_bar(num_timesteps + 1,
                                                              lambda t: math.cos(
                                                                  (t + 0.008) / 1.008 * math.pi / 2) ** 2, ).astype(
                np.float32))

        self.device = device
        self.betas = self.betas.to(device)
        self.alphas = 1.0 - self.betas

        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0).to(self.device)

        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.).to(self.device)

        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5

        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        self.sqrt_inv_alphas_cumprod = torch.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = torch.sqrt(
            1 / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        

        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1).to(x_t.device)
        s2 = s2.reshape(-1, 1).to(x_t.device)

        x0 = s1 * x_t - s2 * noise
        return torch.clamp(x0, min=-1, max=1)

    def q_posterior(self, x_0, x_t, t):
        

        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]

        s1 = s1.reshape(-1, 1).to(x_t.device)
        s2 = s2.reshape(-1, 1).to(x_t.device)
        mu = s1 * x_0 + s2 * x_t
        return mu

    def get_variance(self, t):

        if t == 0:
            return 0

        variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])

        variance = variance.clip(1e-20)
        return variance.to(t.device)

    def step(self,
             model_output,
             timestep,
             sample,
             model_pred_type: str = 'noise'):
        
        t = timestep

        if model_pred_type == 'noise':
            pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        elif model_pred_type == 'x_start':
            pred_original_sample = model_output
        else:
            raise NotImplementedError()

        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = torch.randn_like(model_output)

            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample, pred_original_sample

    def add_noise(self, x_start, x_noise, timesteps):
        timesteps = timesteps.to(self.device)
        s1 = self.sqrt_alphas_cumprod[timesteps]
        s2 = self.sqrt_one_minus_alphas_cumprod[timesteps]

        s1 = s1.reshape(-1, 1).to(x_start.device)
        s2 = s2.reshape(-1, 1).to(x_start.device)
        return s1 * x_start + s2 * x_noise


class MultiModel:
    def __init__(self, xtypes, cfg, device, dims):
        self.decoder = nn.ModuleDict()
        self.encoder = nn.ModuleDict()
        self.ddpm_scheduler = {}
        self.drop_prob = {}
        self.n_T = {}
        self.xtypes = xtypes
        for xtype in xtypes:
            self.encoder[xtype] = Encoder(xtype, dims, cfg[xtype]["encoder"])
            self.decoder[xtype] = Decoder(**cfg[xtype]["decoder"])
            diffcfg = cfg[xtype]["diffusion"]
            self.ddpm_scheduler[xtype] = NoiseScheduler(num_timesteps=diffcfg["n_T"], beta_start=diffcfg["betas"][0],
                                                   beta_end=diffcfg["betas"][1], device=device,
                                                   beta_schedule=diffcfg["schedule"])
            self.drop_prob[xtype] = diffcfg["drop_prob"]
            self.n_T[xtype] = diffcfg["n_T"]


class scBriX(nn.Module):
    def __init__(self, device, cfg, xtypes, n_clusters=28, save_path='./'):
        
        super(scBriX, self).__init__()
        self.xtypes = xtypes
        self.feats_dim = {key: cfg[key]['decoder']['input_size'] for key in cfg}
        model = MultiModel(self.xtypes, cfg, device, self.feats_dim)
        self.encoder = model.encoder
        self.decoder = model.decoder

        self.device = device
        self.n_T = model.n_T
        self.drop_prob = model.drop_prob
        self.mse_loss = nn.MSELoss()
        self.ddpm_scheduler = model.ddpm_scheduler
        self.feats = {}
        self.scaled_feats = {}
        self.svd_model = {}
        self.scaler = {}
        self.freeze = []
        self.lr = None
        self.phases = None
        self.use_cross = False
        self.n_clusters = n_clusters
        self.save_path = save_path
        self.ord = False
        os.makedirs(save_path, exist_ok=True)

    def perturb(self, x, xtype, t=None):
        
        if t is None:
            t = torch.randint(1, self.n_T[xtype] + 1, (x.shape[0],)).to(self.device)
        elif not isinstance(t, torch.Tensor):
            t = torch.tensor([t]).to(self.device).repeat(x.shape[0])

        noise = torch.randn_like(x)
        sche = self.ddpm_scheduler[xtype]
        x_noised = sche.add_noise(x,
                                  noise,
                                  timesteps=t)

        return x_noised, t, noise


    def svd(self, data_dict, mode="train", use_svd=True):

        feats_dim = self.feats_dim
        if not hasattr(self, 'sup_info'):
            self.sup_info = {}

        if len(self.feats.keys()) == len(self.xtypes):
            print(list(data_dict.keys()), "pretrained")
        else:

            for xtype in self.xtypes:
                model_path = os.path.join(self.save_path, f"feats_{xtype}.pt")
                if os.path.isfile(model_path):
                    params = torch.load(model_path)
                    self.svd_model[xtype] = params.get("svd_model", None)
                    self.feats[xtype] = params["feats"]
                    self.scaler[xtype] = params["scaler"]
                    self.scaled_feats[xtype] = params["scaled_feats"]
                    self.sup_info[xtype] = params["sup_info"]

        isreload = len(self.feats.keys()) == len(self.xtypes)
        isBridge = len(self.feats.keys()) > 0 and len(self.feats.keys()) != len(self.xtypes)

        if isreload and mode == "train":
            return


        for xtype, xdata in data_dict.items():
            if mode == "train" and (xtype not in self.feats.keys()):

                sup_info = {}
                if hasattr(xdata, 'obs'):

                    for label_key in ["cell_type_original", 'subclass', 'cell_type', 'label', 'annotation']:

                        if label_key in xdata.obs.columns:
                            sup_info['labels'] = xdata.obs[label_key].astype(str).values

                            print(f"Saved {label_key} labels for {xtype} modality")

                            break

                if xtype == "spatial":


                    if hasattr(xdata, 'obsm') and 'spatial' in xdata.obsm:

                        sup_info['coords'] = xdata.obsm['spatial'].copy()

                        print(f"Saved spatial coordinates from obsm['spatial'], shape: {sup_info['coords'].shape}")

                    elif hasattr(xdata, 'obs'):


                        coord_cols = [c for c in xdata.obs.columns if any(
                            s in c.lower() for s in ['center-x', 'center-y', 'x_coord', "y_coord", "x", "y"])]

                        if len(coord_cols) == 2:
                            sup_info['coords'] = xdata.obs[coord_cols].values

                            print(f"Saved spatial coordinates from obs, using columns: {coord_cols}")


                xdata = xdata.X.toarray() if hasattr(xdata.X, 'toarray') else xdata.X
                xdata = torch.tensor(xdata, dtype=torch.float32)


                if use_svd and xdata.shape[1] > 100:
                    svd = TruncatedSVD(n_components=feats_dim[xtype], random_state=42)
                    feats = svd.fit_transform(xdata)
                    self.svd_model[xtype] = svd
                    self.feats[xtype] = torch.tensor(feats)
                    self.sup_info[xtype] = sup_info
                else:
                    feats = xdata.numpy()
                    self.feats[xtype] = feats
                    self.svd_model[xtype] = None

                    self.sup_info[xtype] = sup_info


            elif mode == "test" or isBridge:

                xdata = xdata.X.toarray() if hasattr(xdata.X, 'toarray') else xdata.X
                xdata = torch.tensor(xdata, dtype=torch.float32)

                if self.svd_model.get(xtype) is not None:
                    feats = self.svd_model[xtype].transform(xdata.numpy())
                else:
                    feats = xdata.numpy()

                self.feats[xtype] = torch.tensor(feats)

            else:
                break

            if mode == "train":
                self.scaler[xtype] = MinMaxScaler()
                feats_scaled = self.scaler[xtype].fit_transform(feats)
                feats_scaled = torch.tensor(feats_scaled) * 2 - 1


                self.scaled_feats[xtype] = feats_scaled

                save_dict = {
                    "feats": self.feats[xtype],
                    "scaled_feats": feats_scaled,
                    "scaler": self.scaler[xtype],
                    "sup_info": self.sup_info[xtype]
                }

                if self.svd_model[xtype] is not None:
                    save_dict["svd_model"] = self.svd_model[xtype]

                torch.save(save_dict, os.path.join(self.save_path, f"feats_{xtype}.pt"))
                print(f"Created features and saved for {xtype}" +
                      (", with spatial metadata" if xtype == "spatial" and xtype in self.sup_info else ""))
            else:
                return torch.tensor(feats)

    def init_cluster_centers(self, data_loader):

        
        kmeans = {}
        y_pred = {}
        features = {xt: [] for xt in self.xtypes}
        n_clusters = self.n_clusters
        with torch.no_grad():
            for x in data_loader:
                for xt in self.xtypes:
                    f, _ = self.encoder[xt](x[xt][0].to(self.device))
                    features[xt].append(f.cpu())

            for xt in self.xtypes:
                features[xt] = torch.cat(features[xt]).numpy()

                kmeans[xt] = KMeans(n_clusters=n_clusters, n_init=20)
                y_pred[xt] = kmeans[xt].fit_predict(features[xt])

        return kmeans, y_pred

    def train_model(self, n_epoch=20, use_cross=False, unpaired=False, match_method='spatial_aware',
                    cell_type_key='label', sampling_strategy='upsample'):
        params = []
        modality_dims = {}
        self.use_cross = use_cross
        self.batch_size = 128
        if not self.feats:
            self.load_feats()

        for xtype, feats in self.feats.items():
            if xtype not in self.freeze:
                params.append({"params": self.decoder[xtype].parameters(), 'lr': self.lr if self.lr else 1e-4})
                params.append({"params": self.encoder[xtype].parameters(), 'lr': self.lr if self.lr else 1e-4})
            else:
                self.decoder[xtype].eval()
                self.encoder[xtype].eval()

            modality_dims[xtype] = 256

        if unpaired:
            train_data = UnpairedMultiModalDataset(self.feats, self.scaled_feats,
                                                   cell_type_key=cell_type_key,
                                                   match_method=match_method,
                                                   sampling_strategy=sampling_strategy,
                                                   sup_info=self.sup_info, n_cluster=self.n_clusters)
        else:
            train_data = MultiModalDataset(self.feats, self.scaled_feats)

        self.to(self.device)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, drop_last=True)

        counter = 0
        min_train_loss = 100

        total_steps = len(train_loader) * n_epoch
        phase_steps = {
            'pretrain': int(0.2 * total_steps),
            'warmup': int(0.2 * total_steps),
            'full': int(0.6 * total_steps)
        }
        early_stop_counter = 200
        self.loss = DynamicHybridLoss(total_steps, phase_steps, self.device, modality_dims, temp=0.1, phase=self.phases,
                                      n_clusters=self.n_clusters)

        params.append({'params': self.loss.parameters(), 'lr': self.lr if self.lr else 1e-5})
        optim = torch.optim.Adam(params, weight_decay=5e-5, lr=self.lr if self.lr else 1e-5)
        pbar = tqdm(range(n_epoch))
        for ep in pbar:

            epoch_total = 0.0
            epoch_diff = 0.0
            epoch_contrast = 0.0
            epoch_intra = 0.0
            epoch_dec = 0.0
            batch_count = 0

            modal_epoch_losses = {xtype: 0.0 for xtype in self.xtypes}


            
            phase = self.loss._get_current_phase()
            weights = self.loss._get_phase_weights(phase)

            if weights["dec"] > 0 and not self.loss.ifdec:
                kmeans, y_pred = self.init_cluster_centers(data_loader=train_loader)
                with torch.no_grad():
                    for mod, kmean in kmeans.items():
                        cluster_centers = torch.tensor(kmean.cluster_centers_,
                                                       dtype=torch.float,
                                                       requires_grad=True,
                                                       device=self.device)
                        self.loss.cluster_centers[mod].copy_(cluster_centers)
                self.loss.ifdec = True

            for xtype in self.xtypes:
                if xtype in self.freeze:
                    self.encoder[xtype].eval()
                    self.decoder[xtype].eval()
                else:
                    self.encoder[xtype].train()
                    self.decoder[xtype].train()
                    self.loss.train()

            for x in train_loader:
                optim.zero_grad()

                total_loss, diffusion_loss, contrastive_loss, iloss, dec_loss, modal_losses = self(x)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters=self.parameters(), max_norm=3)

                optim.step()

                epoch_total += total_loss.item()
                epoch_diff += diffusion_loss.item()
                epoch_contrast += contrastive_loss.item()
                epoch_intra += iloss.item() if isinstance(iloss, torch.Tensor) else iloss
                epoch_dec += dec_loss.item() if isinstance(dec_loss, torch.Tensor) else dec_loss

                for xtype, loss in modal_losses.items():
                    modal_epoch_losses[xtype] += loss.item()

                batch_count += 1


                cur_avg_total = epoch_total / batch_count
                cur_avg_diff = epoch_diff / batch_count
                cur_avg_contrast = epoch_contrast / batch_count
                cur_avg_intra = epoch_intra / batch_count
                cur_avg_dec = epoch_dec / batch_count
                

                modal_loss_str = " | ".join(
                    [f"{xtype}: {(modal_epoch_losses[xtype] / batch_count):.4f}" for xtype in self.xtypes])


                log_str = (f"Phase: {phase} | Loss: {cur_avg_total:.4f} | "
                        # f"rLoss: {cur_avg_diff:.4f} | "
                        f"contrastLoss: {cur_avg_contrast:.4f} | "
                        f"intraLoss: {cur_avg_intra:.4f} | "
                        f"DECLoss: {cur_avg_dec:.4f} | "
                        f"{modal_loss_str} | "
                        f"contrastWeight: {weights['contrast']:.3f} | "
                        f"intraWeight: {weights['intra']:.3f} | "
                        f"decWeight: {weights['dec']:.3f}")
                
                pbar.set_postfix_str(log_str)


            avg_modal_losses = {xtype: loss / batch_count for xtype, loss in modal_epoch_losses.items()}
            non_rna_losses = {key: value for key, value in avg_modal_losses.items() if key != "rna"}
            avg_modal_loss = sum(non_rna_losses.values()) / len(non_rna_losses)

            if avg_modal_loss <= min_train_loss:
                counter = 0
                min_train_loss = avg_modal_loss
                best_t = ep
                for xtype in self.feats:
                    if xtype not in self.freeze:
                        torch.save({
                            f'encoder': self.encoder[xtype].state_dict(),
                            f'decoder': self.decoder[xtype].state_dict()
                        }, os.path.join(self.save_path, f'model_{xtype}.pth'))
            else:
                counter += 1

            if counter >= early_stop_counter:
                print(f'\nEarly stopping at epoch {ep}') 
                break

    def forward(self, x):
        

        z_encodings = {}
        z_e_p = {}

        diffusion_loss = []
        modal_losses = {}
        for i, xtype in enumerate(self.xtypes):
            o_data = [
                (x[other_xtype].to(self.device), other_xtype)
                for other_xtype in self.xtypes
                if other_xtype != xtype
            ] if self.use_cross else None
            z, z_p = self.encoder[xtype](x[xtype][0].to(self.device), o_data)
            z_encodings[xtype] = z
            z_e_p[xtype] = z_p
        z_dict = {xtype: None for xtype in self.xtypes}
        n_mod = len(self.xtypes)

        def compute_loss(target_mod, cond_z):
            x_noised, t, noise = self.perturb(x[target_mod][1].to(self.device), target_mod, t=None)
            mask = self.drop_prob[target_mod]
            pred_noise = self.decoder[target_mod](x_noised, t / self.n_T[target_mod], cond_z, mask)
            return self.mse_loss(noise, pred_noise)

        if "spatial" in self.xtypes or self.ord == True:
            for xtype in self.xtypes:
                nloss = compute_loss(xtype, z_encodings[xtype])
                modal_losses[xtype] = nloss
                diffusion_loss.append(nloss)

        else:
            if n_mod == 2:
                other = [t for t in self.xtypes if t != "rna"][0]
                nloss_rna = compute_loss("rna", z_encodings[other])
                nloss_other = compute_loss(other, z_encodings["rna"])
                modal_losses["rna"] = nloss_rna
                modal_losses[other] = nloss_other
                diffusion_loss.extend([nloss_rna, nloss_other])

            else:
                others = [t for t in self.xtypes if t != "rna"]

                rna_losses = []
                for t in others:
                    rna_losses.append(compute_loss("rna", z_encodings[t]))
                modal_losses["rna"] = sum(rna_losses) / len(rna_losses)
                diffusion_loss.append(modal_losses["rna"])

                for t in others:
                    nloss = compute_loss(t, z_encodings["rna"])
                    modal_losses[t] = nloss
                    diffusion_loss.append(nloss)

        total_loss = self.loss(diffusion_loss, z_e_p, z_encodings)

        diffusion_loss = total_loss["recon"]
        contrastive_loss = total_loss["contrast"]
        iloss = total_loss["intra"]
        dec_loss = total_loss["dec"]

        return total_loss["total"], diffusion_loss, contrastive_loss, iloss, dec_loss, modal_losses

    def load_feats(self, feats_dict={"rna": "feats_rna.pt", "adt": "feats_adt.pt", "atac": "feats_atac.pt"}):
        for xtype, path in feats_dict.items():
            self.feats[xtype] = torch.load(path)["feats"]

    def load_model(self, param_dict=None,
                   freeze=False):
        for xtype, path in param_dict.items():
            loaded_dict = torch.load(path)
            self.encoder[xtype].load_state_dict(loaded_dict['encoder'], strict=True)
            self.decoder[xtype].load_state_dict(loaded_dict['decoder'], strict=True)

            if freeze:
                self.freeze.append(xtype)
                for param in self.encoder[xtype].parameters():
                    param.requires_grad = False
                for param in self.decoder[xtype].parameters():
                    param.requires_grad = False

    def inference(self, out_types=[], condition_data={}, only_decode=False):
        xtypes = self.xtypes
        out_size = 0
        in_types = list(condition_data.keys())
        assert all(key in xtypes for key in in_types), "Not all keys are in the xtypes"
        assert all(key in xtypes for key in out_types), "Not all values in array1 are in xtypes"
        data_obs = None
        for type in in_types:
            out_size = condition_data[type].shape[0]
            data_obs = condition_data[type].obs.copy()
            condition_data[type] = self.svd({type: condition_data[type]}, mode="test")

        z_encodings = {}
        z_dict = {xtype: None for xtype in out_types}

        for xtype, c in condition_data.items():
            other_xtypes = [other for other in in_types if other != xtype]
            if len(other_xtypes) > 0 and self.use_cross:
                o_data = torch.stack([condition_data[other].to(self.device) for other in other_xtypes], dim=1)
            else:
                o_data = None
            encoder = self.encoder[xtype]
            encoder.eval()
            if o_data is not None:
                dataset = TensorDataset(c, o_data)
            else:
                dataset = TensorDataset(c)
            dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
            all_outputs = []
            with torch.no_grad():
                for batch in dataloader:
                    if o_data is not None:
                        c_batch, o_batch = batch
                        c_batch = c_batch.to(self.device)
                        o_batch = o_batch.to(self.device).permute(1, 0, 2)
                        structured_input = [(o, xtype) for xtype, o in zip(other_xtypes, o_batch)]
                        outputs, _ = encoder(c_batch, structured_input)
                    else:
                        c_batch = batch[0].to(self.device)
                        outputs, _ = encoder(c_batch)

                    all_outputs.append(outputs.cpu())

            z_encodings[xtype] = torch.cat(all_outputs, dim=0)

        for xtype in out_types:
            z_others = [z_encodings[t] for t in in_types]
            z_dict[xtype] = sum(z_others)
        output = {}
        for type in out_types:

            if not only_decode:
                noise_scheduler = NoiseScheduler(
                    num_timesteps=1000,
                    beta_start=1e-4,
                    beta_end=0.02,
                    beta_schedule='cosine'
                )
                z = z_dict[type].detach()
                shape = [out_size, self.feats_dim[type]]
                dataset = TensorDataset(torch.zeros(shape).to(self.device), z)
                sample_loader = DataLoader(dataset, batch_size=256, shuffle=False)
                decoder = self.decoder[type].eval()
                imputation = sample_diffusion(decoder,
                                           device=self.device,
                                           dataloader=sample_loader,
                                           noise_scheduler=noise_scheduler,
                                           gt=z,
                                           num_step=1000,
                                           sample_shape=shape,
                                           sample_intermediate=1000,
                                           model_pred_type='noise',
                                           is_classifier_guidance=False,
                                           omega=0.2)
                imp = (imputation + 1) / 2
                imp = imp.detach().cpu().numpy()
                imp = self.scaler[type].inverse_transform(imp)
                # imp = torch.from_numpy(imp)

                if self.svd_model.get(type) is not None:
                    recon = self.svd_model[type].inverse_transform(imp)
                else:
                    recon = imp

            else:
                recon = z_encodings[xtype].detach().cpu().numpy()

            predict = to_adata(recon, type)
            predict.obs = data_obs
            output[type] = predict
        return output
