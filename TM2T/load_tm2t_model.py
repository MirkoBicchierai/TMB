import numpy as np
import torch 
from torch import Tensor
import spacy
from typing import List
from os.path import join as pjoin

from TM2T.utils.word_vectorizer import POS_enumerator
from TM2T.networks.evaluator_wrapper import build_models
from TM2T.utils.get_opt import get_opt
from TM2T.utils.word_vectorizer import WordVectorizer

nlp = spacy.load('en_core_web_sm')


def collate_tensor_with_padding(batch: List[Tensor]) -> Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def process_text(sentence):
    sentence = sentence.replace('-', '')
    doc = nlp(sentence)
    word_list = []
    pos_list = []
    for token in doc:
        word = token.text
        if not word.isalpha():
            continue
        if (token.pos_ == 'NOUN' or token.pos_ == 'VERB') and (word != 'left'):
            word_list.append(token.lemma_)
        else:
            word_list.append(word)
        pos_list.append(token.pos_)
    return word_list, pos_list


def load_tm2t_model_easy(device="cpu", dataset="humanml3d"):
    dataset_opt_path = 'TM2T/checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    opt = get_opt(dataset_opt_path, device)

    opt.dim_word = 300
    opt.max_motion_length = 196
    opt.dim_pos_ohot = len(POS_enumerator)
    opt.dim_motion_hidden = 1024
    opt.max_text_len = 20
    opt.dim_text_hidden = 512
    opt.dim_coemb_hidden = 512
    opt.checkpoints_dir = 'TM2T/checkpoints'

    text_encoder, motion_encoder, movement_encoder = build_models(opt)
    opt = opt
    device = opt.device

    text_encoder.to(opt.device)
    motion_encoder.to(opt.device)
    movement_encoder.to(opt.device)

    text_encoder.eval()
    motion_encoder.eval()
    movement_encoder.eval()

    # Taken from the original code
    motion_mean = np.load(pjoin("TM2T", opt.meta_dir, 'mean.npy'))
    motion_std = np.load(pjoin("TM2T", opt.meta_dir, 'std.npy'))

    w_vectorizer = WordVectorizer('./TM2T/glove', 'our_vab')

    def easy_forward(motions, texts):

        # Text
        cap_lens, word_embs, pos_ohot, tokens_list = [], [], [], []        
        for text in texts:
            word_list, pos_list = process_text(text)
            tokens = ['%s/%s'%(word_list[i], pos_list[i]) for i in range(len(word_list))]
            
            if len(tokens) < opt.max_text_len:
                # pad with "unk"
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                tokens = tokens + ['unk/OTHER'] * (opt.max_text_len + 2 - sent_len)
            else:
                # crop
                tokens = tokens[:opt.max_text_len]
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
                sent_len = len(tokens)
                
            pos_one_hots = []
            word_embeddings = []
            for token in tokens:
                word_emb, pos_oh = w_vectorizer[token]
                pos_one_hots.append(pos_oh[None, :])
                word_embeddings.append(word_emb[None, :])
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)
            word_embeddings = np.concatenate(word_embeddings, axis=0)

            pos_ohot.append(pos_one_hots)
            word_embs.append(word_embeddings)
            cap_lens.append(sent_len)
            tokens_list.append(tokens)

        cap_lens = torch.tensor(cap_lens, device=device)
        word_embs = collate_tensor_with_padding([torch.tensor(w) for w in word_embs])
        pos_ohot = collate_tensor_with_padding([torch.tensor(p) for p in pos_ohot])

        # Motion
        m_lens = torch.tensor([motion.shape[0] for motion in motions], device=device)
        dtype_ = torch.float32
        motions = [(torch.tensor( (motion - motion_mean) / motion_std, dtype=dtype_)) for motion in motions]
        motions = collate_tensor_with_padding([m for m in motions]) # tensor(#B, #F, 263)
        if motions.shape[1] < opt.max_motion_length: # nel paper originale considerano al massimo sequenze lunghe 196
            motions = torch.concatenate([motions,torch.zeros((motions.shape[0], opt.max_motion_length - motions.shape[1], motions.shape[2]))], axis=1)
        
        # non_zero_mask = motions.abs().sum(dim=2) != 0
        # m_lens = non_zero_mask.sum(dim=1)  # shape: (bs,)

        # Now we sort all the tensor based on cap lens otherwise the text_encoder will not work
        sorted_cap_lens, sorted_indices = torch.sort(cap_lens, descending=True)
        word_embs = word_embs[sorted_indices]
        pos_ohot = pos_ohot[sorted_indices]
        motions = motions[sorted_indices]
        m_lens = m_lens[sorted_indices]
        tokens_list = [tokens_list[i] for i in sorted_indices.tolist()]
        cap_lens = sorted_cap_lens  # optional, just to reflect the sorted state

        word_embs = word_embs.detach().to(device).float()
        pos_ohot = pos_ohot.detach().to(device).float()

        align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
        motions = motions[align_idx]
        m_lens = m_lens[align_idx]

        # Motion encoding
        movements = movement_encoder(motions[..., :-4]).detach()
        m_lens = m_lens // opt.unit_length
        motion_embedding = motion_encoder(movements, m_lens)
        motion_latents = motion_embedding # tensor(#B, 512)
        
        # Text encoding
        text_embedding = text_encoder(word_embs, pos_ohot, cap_lens)
        text_embedding = text_embedding[align_idx]
        text_latents = text_embedding 

        return text_latents, motion_latents

    return easy_forward