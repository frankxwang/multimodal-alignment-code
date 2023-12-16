import torch
from torch import nn
from enum import Enum
from main.utils import *
from main.cache import *
from tqdm import tqdm

# scores for the alternate captions
CAPTION_SCORES = "caption_scores"

# alternate captions for a given caption/image pair
ALT_CAPTIONS = "alternate_captions"

class Step:
    def __call__(self, data):
        raise NotImplementedError


class TextScorer(Step, CachedStep):
    
    def __call__(self, data):
        alt_caps = data[ALT_CAPTIONS]
        ragged_alt_caps = RaggedList(alt_caps)
        data[CAPTION_SCORES] = ragged_alt_caps.unflatten(self.score(ragged_alt_caps.flatten()))

    # we use the higher score is better convention
    def score(self, texts):
        raise NotImplementedError


class CausalLLMTextScorer(TextScorer):
    def __init__(self, llm, llm_tokenizer):
        self.llm = llm
        self.llm_tokenizer = llm_tokenizer
        self.init_cache(f"CausalLLMTextScorer__{self.llm.config._name_or_path}")

    @CachedStep.cached_func
    def score(self, texts, batch_size=128):
        scores = []

        for i in range(0, len(texts), batch_size):

            inputs = self.llm_tokenizer(texts[i:i + batch_size], return_tensors='pt', padding=True).to(self.llm.device)
            logits = self.llm(**inputs).logits
            
            # higher score is better, so let's negate this
            avg_losses = -score_logits_sequence(logits, inputs['input_ids'], attn_mask=inputs['attention_mask'], padding_id=self.llm_tokenizer.pad_token_id)

            scores.append(avg_losses)
        
        return torch.cat(scores)


# Generate some set of alternate captions based on the caption
# current convention: include the caption itself since we probably would like to aggregate across many captions
# currently we just have best_k through some scorer or return everything
class AltTextGenerator(CachedStep):
    # have the option of including a TextScorer if text_scorer is not None
    def __init__(self, text_scorer=None, best_k=5):
        self.text_scorer = text_scorer
        self.best_k = best_k
        self.init_cache(f"AltTextGenerator__{text_scorer.llm.config._name_or_path}__best-k-{best_k}")

    @CachedStep.cached_func
    def generate(self, captions):
        alt_texts = self.generate_alt_text(captions)
        if self.text_scorer is not None:
            alt_texts_ragged = RaggedList(alt_texts)
            scores = self.text_scorer.score(alt_texts_ragged.flatten())
            unflat_scores = alt_texts_ragged.unflatten(scores)

            for i, (texts, scores) in enumerate(zip(alt_texts, unflat_scores)):
                scores = torch.tensor(scores)
                alt_texts[i] = [texts[j] for j in torch.argsort(scores)[-self.best_k:].flip(dims=(0,))]

        return alt_texts

    def generate_alt_text(self, captions):
        raise NotImplementedError


class PairSwapsTextGenerator(AltTextGenerator):
    def __init__(self, exclude_words={"the"}, **kwargs):
        super().__init__(**kwargs)
        self.exclude_words = exclude_words

    def generate_alt_text(self, captions):
        results = dict()

        for caption in captions:
            if caption not in results:
                results[caption] = self.generate_alt_text_single(caption)

        return [results[caption] for caption in captions]
    
    def generate_alt_text_single(self, caption):
        space_sep_tokens = caption.split()
        indexes = [i for i in range(len(space_sep_tokens)) if space_sep_tokens[i] not in self.exclude_words]
        
        swaps1 = []
        swaps2 = []
        rearr_sentences = [caption]
        for i in range(len(indexes)):
            for j in range(i + 1, len(indexes)):
                ind1 = indexes[i]
                ind2 = indexes[j]
                if space_sep_tokens[ind1] != space_sep_tokens[ind2]:
                    copy_tokens = list(space_sep_tokens)
                    copy_tokens[ind1], copy_tokens[ind2] = copy_tokens[ind2], copy_tokens[ind1]
                    rearr_sentences.append(" ".join(copy_tokens))
        return rearr_sentences


class ImageTextScorer(CachedStep):
    def score(self, images, texts):
        raise NotImplementedError


class BLIPScoreType(Enum):
    CLM = 1
    ITM = 2
    CONTRASTIVE = 3


# for consistency, higher score = better
class BLIPImageTextScorer(ImageTextScorer):

    def __init__(self, itm_model, itm_processor, score_type=BLIPScoreType.CLM, clm_ignore_sep=True):
        self.itm_model = itm_model
        self.itm_processor = itm_processor
        self.score_type = score_type
        self.clm_ignore_sep = clm_ignore_sep

        # Possible problem: assumes that itm_processor is the same for a given itm_model
        self.init_cache(f"BLIPImageTextScorer__{self.itm_model.config._name_or_path}__{score_type.name}__ignore-sep-{self.clm_ignore_sep}")

    @CachedStep.cached_func
    def score(self, images, texts, batch_size=4):
        assert len(images) == len(texts)
        results = []
        for i in tqdm(range(0, len(images), batch_size)):

            inputs = self.itm_processor(images=images[i:i+batch_size], text=texts[i:i+batch_size], return_tensors="pt", padding=True)
            inputs.to(self.itm_model.device)
    
            if self.score_type == BLIPScoreType.CLM:
    
                lengths = torch.argmin(inputs["attention_mask"], dim=-1) # get first index of 0, if it's 0 this is fine too since we -1
                
                # Don't compute perplexity of the [SEP] token if the flag is on
                if self.clm_ignore_sep:
                    inputs["attention_mask"][torch.arange(len(inputs["attention_mask"])), lengths - 1] = 0
                
                # For ranking purposes, the loss is effectively the same as perplexity.
                # TODO: Maybe check if custom padding tokens might break something here
                results.extend(-score_logits_sequence(self.itm_model(**inputs).logits, inputs["input_ids"], attn_mask=inputs["attention_mask"]))
    
            elif self.score_type == BLIPScoreType.ITM:
    
                results.extend(self.itm_model(**inputs).itm_score[:, 1])
    
            elif self.score_type == BLIPScoreType.CONTRASTIVE:
    
                # for contrastive we get a # of images by # of texts matrix of contrastive scores
                # itm_score actually contains 
                results.extend(torch.diag(self.itm_model(**inputs, use_itm_head=False).itm_score))
        return results


class BLIPAltCaptions(CachedStep):
    def __init__(self, blip_clm_model, blip_clm_processor, num_captions=30, diversity_penalty=0.1, remove_bad_text={"##", "arrafe"}):
        self.blip_clm_model = blip_clm_model
        self.blip_clm_processor = blip_clm_processor
        self.num_captions = num_captions
        self.diversity_penalty = diversity_penalty
        self.remove_bad_text = remove_bad_text

        self.init_cache(f"BLIPAltCaptions__{self.blip_clm_model.config._name_or_path}__num-alt-caps-{self.num_captions}__div-penalty-{self.diversity_penalty}__remove-hashtags-{'-'.join(self.remove_bad_text)}")

    def check_clean(self, text):
        for keyword in self.remove_bad_text:
            if keyword in text:
                return False
        return True

    def generate(self, images, batch_size=8):

        results = []
        for i in range(0, len(images), batch_size):
            inputs = self.blip_clm_processor(images=images[i:i+batch_size], return_tensors="pt").to(self.blip_clm_model.device)

            # Generate multiple captions
            generated_ids = self.blip_clm_model.generate(**inputs, num_return_sequences=self.num_captions, num_beams=self.num_captions, diversity_penalty=self.diversity_penalty, num_beam_groups=self.num_captions)
            
            # Convert generated tokens to text
            captions = [self.blip_clm_processor.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=False) for generated_id in generated_ids]

            assert len(captions) == len(images[i:i+batch_size]) * self.num_captions

            captions_reshaped = [[captions[k] for k in range(j, j+self.num_captions) if self.check_clean(captions[k])] for j in range(0, len(captions), self.num_captions)]

            results.extend(captions_reshaped)
        return results


# class ImageTextScorerAltTextNormed(ImageTextScorer):
#     def __init__(self, image_text_scorer):
#         self.image_text_scorer = image_text_scorer

#     def score(self, images, texts, alt_texts, return_dict=False):
#         orig_scores = self.image_text_scorer.score(images, texts)

#         ragged_alt_texts = RaggedList(alt_texts)

#         flat_texts = ragged_alt_texts.flatten()
#         broadcast_images = ragged_alt_texts.flatten_broadcast(images)

#         mean_scores = torch.tensor(ragged_alt_texts.unflatten(self.image_text_scorer.score(broadcast_images, flat_texts), reduce=torch.mean))

#         normed_scores = orig_scores - mean_scores.to(orig_scores.device)
        
#         if return_dict:
#             return normed_scores, {
#                 "alt_texts": alt_texts,
#                 "orig_scores": orig_scores,
#                 "mean_scores": mean_scores
#             }
        
#         return normed_scores
