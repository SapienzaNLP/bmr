import datetime
from pathlib import Path

import penman
from sacrebleu import corpus_bleu
import torch
from tqdm import tqdm
import smatch
from mspring_bmr.penman import encode

from mspring_bmr.dataset import reverse_direction

def predict_amrs(loader, model, tokenizer, beam_size=1, decoder_start_token_id=0, tokens=None, restore_name_ops=False, return_all=False):

    shuffle_orig = loader.shuffle
    sort_orig = loader.sort

    loader.shuffle = False
    loader.sort = True

    total = len(loader.dataset)
    model.eval()
    model.amr_mode = True

    if tokens is None:
        ids = []
        tokens = []
        with tqdm(total=total) as bar:
            for x, y, extra in loader:
                ii = extra['ids']
                ids.extend(ii)
                with torch.no_grad():
                    out = model.generate(
                        **x,
                        max_length=1024,
                        num_beams=beam_size,
                        # decoder_input_ids=torch.tensor([decoder_start_token_id, 36], device=x['input_ids'].device).repeat(beam_size, 1),
                        forced_bos_token_id=36,
                        num_return_sequences=beam_size,
                        early_stopping=True,
                        no_repeat_ngram_size=0,
                        length_penalty=1.0)


                nseq = len(ii)
                for i1 in range(0, out.size(0), beam_size):
                    tokens_same_source = []
                    tokens.append(tokens_same_source)
                    for i2 in range(i1, i1+beam_size):
                        tokk = out[i2]
                        tokk = [t for t in tokk.tolist()]
                        tokens_same_source.append(tokk)
                
                bar.update(nseq)
        # reorder
        tokens = [tokens[i] for i in ids]
        tokens = [t for tt in tokens for t in tt]


    graphs = []
    for i1 in range(0, len(tokens), beam_size):
        graphs_same_source = []
        graphs.append(graphs_same_source)
        for i2 in range(i1, i1+beam_size):
            tokk = tokens[i2]
            graph, status, (lin, backr) = tokenizer.decode_amr(tokk, restore_name_ops=restore_name_ops)
            graph.status = status
            graph.nodes = lin
            graph.backreferences = backr
            graph.tokens = tokk
            graphs_same_source.append(graph)
        graphs_same_source[:] = tuple(zip(*sorted(enumerate(graphs_same_source), key=lambda x: (x[1].status.value, x[0]))))[1]

    for gps, gg in zip(graphs, loader.dataset.graphs):
        for gp in gps:
            metadata = gg.metadata.copy()
            metadata['annotator'] = 'bart-amr'
            metadata['date'] = str(datetime.datetime.now())
            if 'save-date' in metadata:
                del metadata['save-date']
            gp.metadata = metadata

    loader.shuffle = shuffle_orig
    loader.sort = sort_orig

    if not return_all:
        graphs = [gg[0] for gg in graphs]

    return graphs

def predict_sentences(loader, model, tokenizer, beam_size=1, decoder_start_token_id=0, tokens=None, return_all=False):

    shuffle_orig = loader.shuffle
    sort_orig = loader.sort

    loader.shuffle = False
    loader.sort = True

    total = len(loader.dataset)
    model.eval()
    model.amr_mode = False
    
    if tokens is None:
        ids = []
        tokens = []
        with tqdm(total=total) as bar:
            for x, y, extra in loader:
                ids.extend(extra['ids'])
                x, y = reverse_direction(x, y)
                x['input_ids'] = x['input_ids'][:, :1024]
                x['attention_mask'] = x['attention_mask'][:, :1024]
                with torch.no_grad():
                    out = model.generate(
                        **x,
                        max_length=350,
                        decoder_start_token_id=0,
                        num_beams=beam_size,
                        num_return_sequences=beam_size,
                        early_stopping=False,
                        no_repeat_ngram_size=4,
                        length_penalty=0)
                for i1 in range(0, len(out), beam_size):
                    tokens_same_source = []
                    tokens.append(tokens_same_source)
                    for i2 in range(i1, i1+beam_size):
                        tokk = out[i2]
                        tokk = [t for t in tokk.tolist() if t > 2]
                        tokens_same_source.append(tokk)
                bar.update(out.size(0) // beam_size)
        #reorder
        tokens = [tokens[i] for i in ids]

    sentences = []
    for tokens_same_source in tokens:
        if return_all:
            sentences.append([tokenizer.decode(tokk).strip() for tokk in tokens_same_source])
        else:
            sentences.append(tokenizer.decode(tokens_same_source[0]).strip())

    loader.shuffle = shuffle_orig
    loader.sort = sort_orig

    return sentences


def predict_sentences_multilingual(loader, model, tokenizer, beam_size=1, decoder_start_token_id=0, tokens=None, return_all=False):

    shuffle_orig = loader.shuffle
    sort_orig = loader.sort

    loader.shuffle = False
    loader.sort = True

    total = len(loader.dataset)
    model.eval()
    model.amr_mode = False
    
    if tokens is None:
        ids = []
        tokens = []
        with tqdm(total=total) as bar:
            for x, y, extra in loader:
                ids.extend(extra['ids'])
                x, y = reverse_direction(x, y)
                x['input_ids'] = x['input_ids'][:, :1024]
                x['attention_mask'] = x['attention_mask'][:, :1024]
                with open('test_graph_lorelei.txt', 'a') as f:
                    f.write('\n'.join([str(item) for item in x['input_ids'].tolist()]))
                with torch.no_grad():
                    bad_words = [[bn_id] for bn_id in range(tokenizer.convert_tokens_to_ids("_00046516n"), tokenizer.convert_tokens_to_ids("_00094484v"))] if tokenizer.convert_tokens_to_ids("_00046516n") != tokenizer.unk_token_id else None
                    out = model.generate(
                        **x,
                        max_length=350,
                        decoder_start_token_id=decoder_start_token_id,
                        num_beams=beam_size,
                        num_return_sequences=beam_size,
                        early_stopping= True,
                        # no_repeat_ngram_size= 4,
                        # bad_words_ids=bad_words,
                        length_penalty=1.0)
                        
                for i1 in range(0, len(out), beam_size):
                    tokens_same_source = []
                    tokens.append(tokens_same_source)
                    for i2 in range(i1, i1+beam_size):
                        tokk = out[i2]
                        tokk = [t for t in tokk.tolist() if t > 2 and not ((t >= 250003 ) and (t <= 250025))]
                        tokens_same_source.append(tokk)
                bar.update(out.size(0) // beam_size)
        #reorder
        tokens = [tokens[i] for i in ids]

    sentences = []
    for tokens_same_source in tokens:
        if return_all:
            sentences.append([tokenizer.decode(tokk).strip() for tokk in tokens_same_source])
        else:
            sentences.append(tokenizer.decode(tokens_same_source[0]).strip())

    loader.shuffle = shuffle_orig
    loader.sort = sort_orig

    return sentences

def write_predictions(predictions_path, tokenizer, graphs):
    pieces = [penman.encode(g) for g in graphs]
    Path(predictions_path).write_text('\n\n'.join(pieces).replace(tokenizer.INIT, ''))
    return predictions_path

def compute_smatch(test_path, predictions_path):
    with Path(predictions_path).open() as p, Path(test_path).open() as g:
        score = next(smatch.score_amr_pairs(p, g))

    return score[2]

def compute_smatch_graphs(graphs, test_path, predictions_path, sorted_pred_path):
    with Path(predictions_path).open() as p, Path(test_path).open() as g:
        micro_f_list, score_micro, score = next(score_amr_pairs_with_graphs(graphs, p, g))

    pieces = [encode(g) for g in micro_f_list]
    sorted_pred_path.write_text('\n\n'.join(pieces))

    # print micro and macro score
    print('Micro F1:', score_micro)
    print('Macro F1:', score)

    return score[2]

def compute_bleu(gold_sentences, pred_sentences):
    return corpus_bleu(pred_sentences, [gold_sentences])



def score_amr_pairs_with_graphs(graphs, f1, f2, justinstance=False, justattribute=False, justrelation=False):
    """
    Score one pair of AMR lines at a time from each file handle
    :param f1: file handle (or any iterable of strings) to read AMR 1 lines from
    :param f2: file handle (or any iterable of strings) to read AMR 2 lines from
    :param justinstance: just pay attention to matching instances
    :param justattribute: just pay attention to matching attributes
    :param justrelation: just pay attention to matching relations
    :return: generator of cur_amr1, cur_amr2 pairs: one-line AMR strings
    """
    # matching triple number, triple number in test file, triple number in gold file
    total_match_num = total_test_num = total_gold_num = 0
    micro_f_list = []

    
    # Read amr pairs from two files
    for sent_num, (graph, (cur_amr1, cur_amr2)) in enumerate(zip(graphs, smatch.generate_amr_lines(f1, f2)), start=1):
        best_match_num, test_triple_num, gold_triple_num = smatch.get_amr_match(cur_amr1, cur_amr2,
                                                                         sent_num=sent_num,  # sentence number
                                                                         justinstance=justinstance,
                                                                         justattribute=justattribute,
                                                                         justrelation=justrelation)

        total_match_num += best_match_num
        total_test_num += test_triple_num
        total_gold_num += gold_triple_num
        # clear the matching triple dictionary for the next AMR pair
        smatch.match_triple_dict.clear()
        micro_f_list.append((smatch.compute_f(best_match_num, test_triple_num, gold_triple_num), graph))
        # micro_f_list.append(((best_match_num, test_triple_num, gold_triple_num - best_match_num), graph))
 
    print('total_match_num:', best_match_num)
    print('total_test_num:', test_triple_num)
    print('total_gold_num:', gold_triple_num)
    # calculate mean of micro-f-list
    precission = [micro[0][0] for micro in micro_f_list]
    recall = [micro[0][1] for micro in micro_f_list]
    f_score = [micro[0][2] for micro in micro_f_list]
    mean_precission = sum(precission) / len(precission)
    mean_recall = sum(recall) / len(recall)
    mean_f_score = sum(f_score) / len(f_score)


    for (precision, recall, f_score), graph in micro_f_list:
        graph.metadata['scores'] = "Precision: " + str(precision) + " Recall: " + str(recall) + " F-score: " + str(f_score)


    # sort the micro-f-list by f-score
    micro_f_list = sorted(micro_f_list, key=lambda x: x[0][2], reverse=False)
    micro_f_list = [graph for (precision, recall, f_score), graph in micro_f_list]

    yield micro_f_list, (mean_precission, mean_recall, mean_f_score), smatch.compute_f(total_match_num, total_test_num, total_gold_num)