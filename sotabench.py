import re
from collections import OrderedDict

from sotabencheval.machine_translation import WMTEvaluator, WMTDataset, Language
from fairseq import utils
import models.joint
from tqdm import tqdm
import hubconf


class ModelCfg:
    def __init__(self, model_name, arxiv_id, src_lang, dst_lang, hubname, batch_size, description='', **kwargs):
        self.model_name, self.arxiv_id, self.src_lang, self.dst_lang = model_name, arxiv_id, src_lang, dst_lang
        self.hubname, self.batch_size = hubname, batch_size
        self.params = kwargs
        if self.params.get('tokenizer') == 'moses':
            self.params.setdefault('moses_no_dash_splits', True)
            self.params.setdefault('moses_no_escape', False)

        self.description = self._get_description(description)

    def _get_description(self, description):
        details = []
        if description:
            details.append(description)

        ensemble_len = len(self.params.get('checkpoint_file', '').split(':'))
        if ensemble_len > 1:
            details.append('ensemble of {} models'.format(ensemble_len))
        details.append('batch size: {}'.format(self.batch_size))
        details.append('beam width: {}'.format(self.params['beam']))
        lenpen = self.params.get('lenpen', 1)
        if lenpen != 1:
            details.append('length penalty: {:.2f}'.format(lenpen))
        return ', '.join(details)

    def get_evaluator(self, model, dataset):
        def tok4bleu(sentence):
            tokenized = model.tokenize(sentence)
            return re.sub(r'(\S)-(\S)', r'\1 ##AT##-##AT## \2', tokenized)

        return WMTEvaluator(
            dataset=dataset,
            source_lang=self.src_lang,
            target_lang=self.dst_lang,
            local_root="data/nlp/wmt",
            model_name=self.model_name,
            paper_arxiv_id=self.arxiv_id,
            model_description=self.description,
            tokenization=tok4bleu
        )

    def load_model(self):
        # similar to torch.hub.load, but makes sure to load hubconf from the current commit
        load = getattr(hubconf, self.hubname)
        return load(**self.params).cuda()


def translate_batch(model, sids, sentences):
    input = [model.encode(sentence) for sentence in sentences]
    lengths = [len(t) for t in input]
    dataset = model.task.build_dataset_for_inference(input, lengths)
    samples = dataset.collater(dataset)
    samples = utils.apply_to_sample(
        lambda tensor: tensor.to(model.device),
        samples
    )
    ids = samples['id'].cpu()

    generator = model.task.build_generator(model.args)

    translations = model.task.inference_step(generator, model.models, samples)
    hypos = [translation[0]['tokens'] for translation in translations]
    translated = [model.decode(hypo) for hypo in hypos]
    return OrderedDict([(sids[id], tr) for id, tr in zip(ids, translated)])


def batchify(items, batch_size):
    items = list(items)
    items = sorted(items, key=lambda x: len(x[1]), reverse=True)
    length = len(items)
    return [items[i * batch_size: (i+1) * batch_size] for i in range((length + batch_size - 1) // batch_size)]


datasets = [
    (WMTDataset.News2014, Language.English, Language.German),
    (WMTDataset.News2014, Language.English, Language.French),
    (WMTDataset.News2019, Language.English, Language.German),
]

models = [
    # English -> German models
    ModelCfg("Local Joint Self-attention", "1905.06596", Language.English, Language.German, 'joint_attention.wmt16.en-de',
             description="trained on WMT16",
             batch_size=64, beam=5, tokenizer='moses', bpe='subword_nmt', lenpen=0.35),

    # English -> French models
    ModelCfg("Local Joint Self-attention", "1905.06596", Language.English, Language.French, 'joint_attention.wmt14.en-fr',
             description="trained on WMT14",
             batch_size=64, beam=5, tokenizer='moses', bpe='subword_nmt', lenpen=0.9),
]

for model_cfg in models:
    print("Evaluating model {} ({} -> {})".
          format(model_cfg.model_name, model_cfg.src_lang.name, model_cfg.dst_lang.name))
    model = model_cfg.load_model()
    for ds, src_lang, dst_lang in datasets:
        if src_lang == model_cfg.src_lang and dst_lang == model_cfg.dst_lang:
            evaluator = model_cfg.get_evaluator(model, ds)

            with tqdm(batchify(evaluator.metrics.source_segments.items(), model_cfg.batch_size)) as iter:
                for batch in iter:
                    sids, texts = zip(*batch)
                    answers = translate_batch(model, sids, texts)
                    evaluator.add(answers)
                    if evaluator.cache_exists:
                        break

            evaluator.save()
            print(evaluator.results)
