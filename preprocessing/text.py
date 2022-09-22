import tensorflow as tf
import numpy as np
import tensorflow_unicorn as tfu
import unicodedata
import abc
from collections import defaultdict

_PAD = "[PAD]"
_CLS = "[CLS]"
_SEP = "[SEP]"
_MASK = "[MASK]"
_UNK = "[UNK]"
_RESTRICTED_VOCABS = [
    _PAD, _CLS, _SEP, _MASK, _UNK
]


def _is_special(word):
    return word.startswith("[") and word.endswith("]")


def _remove_prefix(text, prefix):
    return text[len(prefix):]


class Tokenizer:
    """Abstract class for tokenizers.
    """
    @abc.abstractmethod
    def _tokenize(self, sentence):
        raise NotImplementedError("Abstract method")

    def tokenize(self, inputs):
        if isinstance(inputs, str):
            inputs = unicodedata.normalize('NFD', inputs)
            return self._tokenize(inputs)
        else:
            res = []
            for sentence in inputs:
                res.append(self._tokenize(sentence))
            return res

    @abc.abstractmethod
    def _detokenize(self, tokens):
        raise NotImplementedError("Abstract method")

    def detokenize(self, inputs):
        if np.ndim(inputs[0]) == 0:
            return self._detokenize(inputs)
        else:
            res = []
            for tokens in inputs:
                res.append(self._detokenize(tokens))
            return res


class BasicTokenizer(Tokenizer):
    def __init__(self, vocab_path, lower_case=True, encoding="utf-8"):
        self._lower_case = lower_case
        # self._splitter = re.compile(r"\[[A-Za-z]+\]|[^A-Za-z0-9_\- ]|[A-Za-z0-9_-]+")
        self._load_vocabulary(vocab_path, encoding)
        
    def _load_vocabulary(self, vocab_path, encoding):
        self.vocab_dict = defaultdict(lambda: self.unk_token)
        self.token_dict = []
        for i, vocab in enumerate(tfu.utils.io.text_line_generator(vocab_path, encoding=encoding, strip=False)):
            self.vocab_dict[vocab] = i
            self.token_dict.append(vocab)

        self.unk_token = self.vocab_dict[_UNK]
        self.cls_token = self.vocab_dict[_CLS]
        self.sep_token = self.vocab_dict[_SEP]
        self.pad_token = self.vocab_dict[_PAD]
        self.mask_token = self.vocab_dict[_MASK]

        self.vocab_size = len(self.vocab_dict)

    def word2token(self, word):
        return [self.vocab_dict.get(word, self.unk_token)]

    def _tokenize(self, sentence):
        assert isinstance(sentence, str)
        if self._lower_case:
            sentence = sentence.lower()
        res = []
        for vocab in self._split(sentence):
            res.extend(self.word2token(vocab))
        return res

    def _split(self, sentence):
        start, end = 0, 0
        length = len(sentence)
        while end < length:
            char = sentence[end]
            if self._is_cjk_char(char) or self._is_punctuation(char):
                if start < end:
                    yield sentence[start:end]
                yield sentence[end]
                end += 1
                start = end
            elif self._is_blank(char):
                if start < end:
                    yield sentence[start:end]
                end += 1
                start = end
            else:
                end += 1
        if start < end:
            yield sentence[start:end]

    def _detokenize(self, tokens):
        res = []
        for token in tokens:
            res.append(self.token_dict[token])
        return ' '.join(res)

    @staticmethod
    def _is_chinese(char):
        return '\u4E00' <= char <= '\u9FFF'

    @staticmethod
    def _is_cjk_char(char):
        cp = ord(char)
        return \
            0x4E00 <= cp <= 0x9FFF or \
            0x3400 <= cp <= 0x4DBF or \
            0x20000 <= cp <= 0x2A6DF or \
            0x2A700 <= cp <= 0x2B73F or \
            0x2B740 <= cp <= 0x2B81F or \
            0x2B820 <= cp <= 0x2CEAF or \
            0xF900 <= cp <= 0xFAFF or \
            0x2F800 <= cp <= 0x2FA1F

    @staticmethod
    def _is_punctuation(char):
        # return unicodedata.category(char)[0] == 'P'
        cp = ord(char)
        return \
            33 <= cp <= 47 or \
            58 <= cp <= 64 or \
            91 <= cp <= 96 or \
            123 <= cp <= 126 or \
            unicodedata.category(char).startswith('P')

    @staticmethod
    def _is_blank(char):
        return char == ' ' or char == '\t' or char == '\n' or char == '\r' or unicodedata.category(char) == 'Zs'


class WordpieceTokenizer(BasicTokenizer):
    def __init__(self, vocab_path, suffix_indicator="##", **kwargs):
        self._suffix_indicator = suffix_indicator
        super(WordpieceTokenizer, self).__init__(vocab_path, **kwargs)
        self._load_wordpieces()

    def _load_wordpieces(self):
        self._prefixes = {}
        self._suffixes = {}
        for vocab in self.vocab_dict:
            if _is_special(vocab):
                pass
            elif vocab.startswith(self._suffix_indicator):
                self._suffixes[_remove_prefix(vocab, self._suffix_indicator)] = self.vocab_dict[vocab]
            else:
                self._prefixes[vocab] = self.vocab_dict[vocab]

    def word2token(self, word):
        if word in self._prefixes:
            return [self._prefixes[word]]

        length = len(word)
        start, end = 0, length
        res = []
        while start < length:
            while end > start:
                if start == 0 and word[start:end] in self._prefixes:
                    res.append(self._prefixes[word[start:end]])
                elif word[start:end] in self._suffixes:
                    res.append(self._suffixes[word[start:end]])
                    break
                end -= 1
            else:
                res.append(self.unk_token)
                break
            start = end
            end = length
        return res

    def _detokenize(self, tokens):
        words = []
        for token in tokens:
            word = self.token_dict[token]
            if word.startswith(self._suffix_indicator):
                words[-1] += _remove_prefix(word, self._suffix_indicator)
            else:
                words.append(word)
        return ' '.join(words)


class TwoStageTokenizer:
    def __init__(self, tokenizer, processor):
        self.tokenizer = tokenizer
        self.processor = processor

    def tokenize(self, inputs):
        return np.array(self.tokenizer.tokenize(inputs), dtype=np.int32)

    def detokenize(self, inputs):
        return self.tokenizer.detokenize(inputs)

    def encode(self, text1, text2=None):
        tokens1 = self.tokenize(text1)
        if text2 is None:
            tokens2 = None
        else:
            tokens2 = self.tokenize(text2)
        return self.processor.trim(tokens1, tokens2)

    def gen_masked_lm_input(self, text1, text2=None,
                            remain_token_limit=None, vocab_size=None,
                            mask_rate=0.15, mask_mask_rate=0.8,
                            mask_random_rate=0.1, return_mlm_mask=False):
        return self.processor.gen_masked_lm_data(
            self.encode(text1, text2),
            remain_token_limit, vocab_size,
            mask_rate, mask_mask_rate,
            mask_random_rate, return_mlm_mask
        )


class TFBertTokenPostProcessor:
    def __init__(self, cls_token, sep_token, mask_token, seq_len=512,
                 remain_token_limit=None, vocab_size=None,
                 truncate_strategy="last"):
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.mask_token = mask_token
        self.seq_len = seq_len
        self._remain_token_limit = remain_token_limit
        self._vocab_size = vocab_size
        self._trim_strategy = truncate_strategy

        self._FIRST_SENTENCE_SIZE = self._get_trimmer(seq_len, truncate_strategy)

    @tf.function
    def single_trim(self, tokens1, tokens2=None):
        CLS_TOKEN = tf.constant([self.cls_token])
        SEP_TOKEN = tf.constant([self.sep_token])

        if tokens2 is None:
            tokens_input = tf.concat(
                [
                    CLS_TOKEN,
                    tokens1[:self.seq_len-2],
                    SEP_TOKEN
                ],
                axis=-1
            )
            padding = [[0, tf.maximum(0, self.seq_len - tf.shape(tokens_input)[0])]]
            tokens_input = tf.pad(tokens_input, padding)
            segment_input = tf.zeros_like(tokens_input, dtype=tf.int32)
            return tokens_input, segment_input
        else:
            tokens_input = tf.concat(
                [
                    CLS_TOKEN,
                    tokens1[:self._FIRST_SENTENCE_SIZE],
                    SEP_TOKEN
                ],
                axis=-1
            )
            segment_input = tf.zeros_like(tokens_input, dtype=tf.int32)
            padding = [[0, tf.maximum(0, self.seq_len - tf.shape(tokens_input)[0])]]
            segment_input = tf.pad(segment_input, padding, constant_values=1)
            tokens_input = tf.concat([tokens_input, tokens2], axis=-1)[:self.seq_len-1]
            tokens_input = tf.concat([tokens_input, SEP_TOKEN], axis=-1)
            padding = [[0, tf.maximum(0, self.seq_len - tf.shape(tokens_input)[0])]]
            tokens_input = tf.pad(tokens_input, padding)
            return tokens_input, segment_input

    @tf.function
    def batch_trim(self, tokens1, tokens2=None):
        CLS_TOKEN = tf.ones_like(tokens1[:, :1]) * self.cls_token
        SEP_TOKEN = tf.ones_like(tokens1[:, :1]) * self.sep_token

        if tokens2 is None:
            tokens_input = tf.concat(
                [
                    CLS_TOKEN,
                    tokens1[..., :self.seq_len-2],
                    SEP_TOKEN
                ],
                axis=-1
            )
            tokens_input = tokens_input.to_tensor(shape=(None, self.seq_len), default_value=0)
            segment_input = tf.zeros_like(tokens_input, dtype=tf.int32)
            return tokens_input, segment_input
        else:
            tokens_input = tf.concat(
                [
                    CLS_TOKEN,
                    tokens1[..., :self._FIRST_SENTENCE_SIZE],
                    SEP_TOKEN
                ],
                axis=-1
            )
            segment_input = tf.zeros_like(tokens_input, dtype=tf.int32)
            tokens_input = tf.concat([tokens_input, tokens2], axis=-1)[:self.seq_len-1]
            tokens_input = tf.concat([tokens_input, SEP_TOKEN], axis=-1)
            tokens_input = tokens_input.to_tensor(shape=(None, self.seq_len), default_value=0)
            segment_input = segment_input.to_tensor(shape=(None, self.seq_len), default_value=1)
            return tokens_input, segment_input

    def trim(self, tokens1, tokens2=None):
        if not isinstance(tokens1, tf.RaggedTensor):
            tokens1 = tf.ragged.constant(tokens1)
        if not isinstance(tokens2, tf.RaggedTensor) and tokens2 is not None:
            tokens2 = tf.ragged.constant(tokens2)

        if tf.rank(tokens1) == 1:
            return self.single_trim(tokens1, tokens2)
        return self.batch_trim(tokens1, tokens2)

    @staticmethod
    def _get_trimmer(seq_len, trim_strategy):
        if trim_strategy == "last":
            return seq_len - 3
        elif trim_strategy == "equal":
            return (seq_len - 3) // 2

    @tf.function
    def gen_masked_lm_data(self, inputs,
                           remain_token_limit=None, vocab_size=None,
                           mask_rate=0.15, mask_mask_rate=0.8,
                           mask_random_rate=0.1, return_mlm_mask=False):
        if remain_token_limit is None:
            if self._remain_token_limit is None:
                raise ValueError("arguments remain_token_limit should be specified in __init__ or gen_masked_lm_input")
            remain_token_limit = self._remain_token_limit
        if vocab_size is None:
            if self._vocab_size is None:
                raise ValueError("arguments remain_token_limit should be specified in __init__ or gen_masked_lm_input")
            vocab_size = self._vocab_size

        mlm_output, segment_ids = inputs

        token_ids, mask = self._masking(
            mlm_output, remain_token_limit, vocab_size,
            mask_rate, mask_mask_rate,
            mask_random_rate
        )

        if return_mlm_mask:
            return (token_ids, segment_ids, mask), mlm_output
        else:
            return (token_ids, segment_ids), mlm_output

    def _masking(self, token_ids,
                 remain_token_limit, vocab_size,
                 mask_rate=0.15, mask_mask_rate=0.8,
                 mask_random_rate=0.1):

        # while (some rows do not have masks)
        mlm_mask = tf.zeros_like(token_ids, dtype=tf.bool)
        available_pos = tf.where(token_ids >= remain_token_limit)
        while tf.reduce_any(tf.logical_not(tf.reduce_any(mlm_mask, axis=-1))):
            mlm_mask = tf.tensor_scatter_nd_update(
                mlm_mask, available_pos,
                tf.random.uniform((tf.shape(available_pos)[0],), 0, 1) < mask_rate
            )

        mask_pos = tf.where(mlm_mask)
        r = tf.random.uniform((tf.shape(mask_pos)[0],), 0, 1)
        mask_mask_pos = tf.gather_nd(
            mask_pos, tf.where(r < mask_mask_rate)
        )
        token_ids = tf.tensor_scatter_nd_update(
            token_ids, mask_mask_pos,
            tf.repeat(tf.constant([self.mask_token], dtype=tf.int32), tf.shape(mask_mask_pos)[0])
        )
        mask_random_pos = tf.gather(
            mask_pos, tf.where(tf.logical_and(r >= mask_mask_rate, r < mask_mask_rate + mask_random_rate))[:, 0]
        )
        token_ids = tf.tensor_scatter_nd_update(
            token_ids, mask_random_pos,
            tf.random.uniform((tf.shape(mask_random_pos)[0],), remain_token_limit, vocab_size, dtype=tf.int32)
        )
        return token_ids, mlm_mask

    @classmethod
    def from_tokenizer(cls, tokenizer, seq_len=None, truncate_strategy=None):
        config = {
            'remain_token_limit': 0,
            'vocab_size': tokenizer.vocab_size,
        }
        if seq_len is not None:
            config['seq_len'] = seq_len
        if truncate_strategy is not None:
            config['truncate_strategy'] = truncate_strategy

        for word in tokenizer.vocab_dict:
            if _is_special(word):
                config['remain_token_limit'] += 1
                if word == _CLS:
                    config['cls_token'] = tokenizer.vocab_dict[word]
                elif word == _SEP:
                    config['sep_token'] = tokenizer.vocab_dict[word]
                elif word == _MASK:
                    config['mask_token'] = tokenizer.vocab_dict[word]
            else:
                break
        return cls(**config)

    @classmethod
    def from_vocab_file(cls, vocab_file, encoding="utf-8", seq_len=None, truncate_strategy=None):
        config = {
            'remain_token_limit': 0,
            'vocab_size': 0,
        }
        if seq_len is not None:
            config['seq_len'] = seq_len
        if truncate_strategy is not None:
            config['truncate_strategy'] = truncate_strategy

        FINDING_SPECIAL = True
        for i, word in enumerate(tfu.utils.io.text_line_generator(vocab_file, encoding=encoding, strip=False)):
            config['vocab_size'] += 1
            if FINDING_SPECIAL:
                if _is_special(word):
                    config['remain_token_limit'] += 1
                    if word == _CLS:
                        config['cls_token'] = i
                    elif word == _SEP:
                        config['sep_token'] = i
                    elif word == _MASK:
                        config['mask_token'] = i
                else:
                    FINDING_SPECIAL = False

        return cls(**config)


class TFTokenizer:
    """Abstract class for tokenizers in graph mode.
    """
    @abc.abstractmethod
    def _tokenize(self, inputs):
        raise NotImplementedError("Abstract method")

    def tokenize(self, inputs):
        return self._tokenize(inputs)

    @abc.abstractmethod
    def _detokenize(self, inputs):
        raise NotImplementedError("Abstract method")

    def detokenize(self, inputs):
        return self._detokenize(inputs)


class TFBasicTokenizer(TFTokenizer):
    """Basic tokenizer.
    """
    def __init__(self, vocab_path, unk_token, lower_case=True, encoding="utf-8"):
        self._lower_case = lower_case
        self._encoding = encoding

        init = tf.lookup.TextFileInitializer(
            filename=vocab_path,
            key_dtype=tf.string, key_index=tf.lookup.TextFileIndex.WHOLE_LINE,
            value_dtype=tf.int64, value_index=tf.lookup.TextFileIndex.LINE_NUMBER,
        )
        self._vocab_lookup_table = tf.lookup.StaticHashTable(init, unk_token)
        self.vocab_size = self._vocab_lookup_table.size().numpy()

        init = tf.lookup.TextFileInitializer(
            filename=vocab_path,
            key_dtype=tf.int64, key_index=tf.lookup.TextFileIndex.LINE_NUMBER,
            value_dtype=tf.string, value_index=tf.lookup.TextFileIndex.WHOLE_LINE,
        )
        self._detokenize_table = tf.lookup.StaticHashTable(init, _UNK)
        self._splitter = r"|[[:^word:]]|[[:word:]]+"

    def _tokenize(self, inputs):
        words = tf.strings.regex_replace(
            inputs, self._splitter, r" \0",
            name="regex_formatter",
        )
        tokens = self._vocab_lookup_table.lookup(
            tf.strings.split(words),
            name="word_lookup"
        )
        return tokens

    def detokenize(self, inputs):
        words = self._detokenize_table.lookup(inputs)
        sentence = tf.strings.reduce_join(words, axis=-1, separator=' ')
        return sentence
