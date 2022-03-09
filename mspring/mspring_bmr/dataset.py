import logging
import random
import torch
from cached_property import cached_property
from torch.utils.data import Dataset
from mspring_bmr.IO import read_raw_amr_data, read_amr_data

def reverse_direction(x, y, pad_token_id=1):
    input_ids = torch.cat([y['decoder_input_ids'], y['labels'][:, -1:]], 1)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[input_ids == pad_token_id] = 0
    decoder_input_ids = x['input_ids'][:,:-1]
    lm_labels = x['input_ids'][:,1:]
    x = {'input_ids': input_ids, 'attention_mask': attention_mask}
    y = {'decoder_input_ids': decoder_input_ids, 'labels': lm_labels}
    return x, y

lang_map = {"en": "en_XX", "it": "it_IT", "es": "es_XX", "fr": "fr_FR", "de": "de_DE", "pt": "pt_PT", "zh": "zh_CN"}
class AMRDataset(Dataset):
    
    def __init__(
        self,
        paths,
        tokenizer,
        snt_tokenizer,
        device=torch.device('cpu'),
        use_recategorization=False,
        remove_longer_than=None,
        remove_wiki=False,
        dereify=True,
        raw_data=True
    ):
        self.paths = paths
        self.tokenizer = tokenizer
        self.snt_tokenizer = snt_tokenizer
        self.device = device

        if raw_data:
            graphs = read_raw_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        else:
            graphs = read_amr_data(paths, use_recategorization, remove_wiki=remove_wiki, dereify=dereify)
        self.graphs = []
        self.sentences = []
        self.linearized = []
        self.linearized_extra = []
        self.tok_snts = []
        self.remove_longer_than = remove_longer_than
        self.snt_lang = []
        self.doc = []
        self.dif_doc = []

        for g in graphs:
            l, e = self.tokenizer.linearize(g)
  
            try:
                # tok_snt = self.snt_tokenizer.tokenize(g.metadata['snt'])
                tok_snt = len(g.metadata['snt'].split())
            except:
                logging.warning('Invalid sentence!')
                continue

            if remove_longer_than and len(l) > remove_longer_than:
                continue
            if len(l) > 1024:
                logging.warning('Sequence longer than 1024 included. BART does not support it!')
            
            self.tok_snts.append(tok_snt)
            self.sentences.append(g.metadata['snt'])
            
            if 'lng' in g.metadata:
                self.snt_lang.append(lang_map[g.metadata['lng']])
            else:
                self.snt_lang.append(None)

            if 'doc' in g.metadata:
                doc_token = '<' + g.metadata['doc'].strip() + '>'
                self.doc.append(doc_token)
                if doc_token not in self.dif_doc:
                    self.dif_doc.append(doc_token)

            else:
                self.doc.append(None)

            self.graphs.append(g)
            self.linearized.append(l)
            self.linearized_extra.append(e)

        self.snt_tokenizer.add_tokens(self.dif_doc)
        self.snt_tokenizer.delete_tokens = self.dif_doc
        self.tokenizer.add_tokens(self.dif_doc)
        self.tokenizer.delete_tokens = self.dif_doc

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sample = {}
        sample['id'] = idx
        sample['sentences'] = self.sentences[idx]
        sample['lng'] = self.snt_lang[idx]
        sample['graphs'] = self.graphs[idx]
        sample['tok_sentences'] = self.tok_snts[idx]
        sample['doc'] = self.doc[idx]
        if self.linearized is not None:
            sample['linearized_graphs_ids'] = self.linearized[idx]
            sample.update(self.linearized_extra[idx])            
        return sample
    
    def size(self, sample):
        return len(sample['sentences'])
    
    def collate_fn(self, samples, device=torch.device('cpu')):
        x = [s['sentences'] for s in samples]
        x_lng = [s['lng'] for s in samples]
        x_doc = [s['doc'] for s in samples]

        x, extra = self.batch_encode_sentences(x, x_lng, x_doc, device=device)

        y = [s['graphs'] for s in samples]
        y, extra_y = self.tokenizer.batch_encode_graphs(y, device=device)
        extra.update(extra_y)

        if 'linearized_graphs_ids' in samples[0]:
            y = [s['linearized_graphs_ids'] for s in samples]
            y, extra_y = self.tokenizer.batch_encode_graphs_from_linearized(y, samples, device=device)
            extra.update(extra_y)

            if None not in x_doc:
                # add sentence doc after begining of sentence token
                sentences_doc_tensor = torch.tensor(self.snt_tokenizer.convert_tokens_to_ids(x_doc), dtype=torch.long, device=y['decoder_input_ids'].device)
                
                # concat begining of sentence token in input_ids
                y["decoder_input_ids"] = torch.cat((y["decoder_input_ids"][:, :1], sentences_doc_tensor.unsqueeze(1), y["decoder_input_ids"][:, 1:]), 1)
                y["labels"] = torch.cat((sentences_doc_tensor.unsqueeze(1), y["labels"][:, :]), 1)
        else:
            y = None

        extra['ids'] = [s['id'] for s in samples]
        return x, y, extra

    def batch_encode_sentences(self, sentences, sentences_lng, sentences_doc, device=torch.device('cpu')):
        sentences = [s for s in sentences]
        extra = {'sentences': sentences}

        batch = self.snt_tokenizer.batch_encode_plus(sentences, return_tensors='pt', padding=True)

        if None not in sentences_lng:
            sentences_lng_tensor = torch.tensor(self.snt_tokenizer.convert_tokens_to_ids(sentences_lng), dtype=torch.long, device=batch['input_ids'].device)

            if self.tokenizer.direction == "text":
                batch["input_ids"][batch["input_ids"]==self.snt_tokenizer.convert_tokens_to_ids(self.snt_tokenizer.src_lang)] = 1
                batch["input_ids"] = torch.roll(batch["input_ids"], 1, 1)
                batch["input_ids"][:, 0] = sentences_lng_tensor

            else:
                batch["input_ids"][batch["input_ids"]==self.snt_tokenizer.convert_tokens_to_ids(self.snt_tokenizer.src_lang)] = sentences_lng_tensor
        

        if None not in sentences_doc:
            # add sentence doc after begining of sentence token
            sentences_doc_tensor = torch.tensor(self.snt_tokenizer.convert_tokens_to_ids(sentences_doc), dtype=torch.long, device=batch['input_ids'].device)
            
            # concat begining of sentence token in input_ids
            batch["input_ids"] = torch.cat((batch["input_ids"][:, :1], sentences_doc_tensor.unsqueeze(1), batch["input_ids"][:, 1:]), 1)

            # add 1 in attention_mask
            batch["attention_mask"] = torch.cat((torch.ones_like(sentences_doc_tensor).unsqueeze(1), batch["attention_mask"]), 1)

        batch = {k: v.to(device) for k, v in batch.items()}
        return batch, extra

class AMRDatasetTokenBatcherAndLoader:
    
    def __init__(self, dataset, batch_size=800 ,device=torch.device('cpu'), shuffle=False, sort=False):
        assert not (shuffle and sort)
        self.batch_size = batch_size
        self.tokenizer = dataset.tokenizer
        self.dataset = dataset
        self.device = device
        self.shuffle = shuffle
        self.sort = sort

    def __iter__(self):
        it = self.sampler()
        it = ([[self.dataset[s] for s in b] for b in it])
        it = (self.dataset.collate_fn(b, device=self.device) for b in it)
        return it

    @cached_property
    def sort_ids(self):
        lengths = [len(s.split()) for s in self.dataset.sentences]
        ids, _ = zip(*sorted(enumerate(lengths), reverse=True))
        ids = list(ids)
        return ids

    def sampler(self):
        ids = list(range(len(self.dataset)))[::-1]
        
        if self.shuffle:
            random.shuffle(ids)
        if self.sort:
            ids = self.sort_ids.copy()

        batch_longest = 0
        batch_nexamps = 0
        batch_ntokens = 0
        batch_ids = []

        def discharge():
            nonlocal batch_longest
            nonlocal batch_nexamps
            nonlocal batch_ntokens
            ret = batch_ids.copy()
            batch_longest *= 0
            batch_nexamps *= 0
            batch_ntokens *= 0
            batch_ids[:] = []
            return ret

        while ids:
            idx = ids.pop()
            size = self.dataset.size(self.dataset[idx])
            cand_batch_ntokens = max(size, batch_longest) * (batch_nexamps + 1)
            if cand_batch_ntokens > self.batch_size and batch_ids:
                yield discharge()
            batch_longest = max(batch_longest, size)
            batch_nexamps += 1
            batch_ntokens = batch_longest * batch_nexamps
            batch_ids.append(idx)

            if len(batch_ids) == 1 and batch_ntokens > self.batch_size:
                yield discharge()

        if batch_ids:
            yield discharge()
