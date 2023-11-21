import typing as T
from functools import lru_cache
from pathlib import Path
import glob
import pickle

import more_itertools as mit
import torch
import clip
import numpy as np
import PIL.Image as pil
from sklearn.manifold import TSNE
from tqdm import tqdm


def _batch_encode(model, corpus):
    res = []
    batch_size = 30
    dataset = mit.chunked(corpus, batch_size)
    iterations = len(corpus) / 30
    for batch in tqdm(dataset, total=iterations):
        batch = torch.stack(batch)
        res.extend(model.encode_text(batch))
    return torch.stack(res)


def load_corpus(fpath: str) -> T.List[str]:
    lines = []
    with open(fpath, 'r') as f:
        for li in f.readlines():
            li = li.replace('\n', '').strip()
            if (li != '' and li[0] != '#'):
                lines.append(li)
        return lines


def grep_images(root_path: T.Union[Path, str], recursive=False) -> T.List[Path]:
    if isinstance(root_path, Path):
        root_str = str(root_path.absolute())
    else:
        root_str = root_path

    images_fpaths = []
    if recursive:
        root_str = str(root_str) + '/**'

    for suff in ['.JPEG', '.JPG', '.jpeg', '.jpg', '.png', '.tif']:
        suff_images = glob.glob(f'{root_str}/*{suff}', recursive=True)
        images_fpaths.extend(suff_images)
    images_fpaths = [Path(x) for x in images_fpaths]
    images_fpaths = sorted(images_fpaths, key=lambda x: x.name)

    return images_fpaths


def _get_features(
    corpus: T.List[str] = [],
    images: T.List[Path] = []
) -> T.Tuple[
    np.ndarray,
    T.List[T.Union[str, Path]]
]:
    if (len(images) == 0) and (len(corpus) == 0):
        raise ValueError('No valid arguments provided')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-B/32', device=device)

    image_features = []
    if len(images) > 0:
        pil_images = [pil.open(x) for x in images]
        for img in pil_images:
            preproc_image = preprocess(img).unsqueeze(0).to(device)  # type: ignore
            img_features = model.encode_image(preproc_image)
            img_features = img_features / img_features.norm(dim=1, keepdim=True)
            image_features.append(img_features.cpu().numpy())
        image_features = np.concatenate(image_features)

    corpus_features = []
    if len(corpus) > 0:
        tokenized_corpus = clip.tokenize(corpus).to(device)
        corpus_features = _batch_encode(model, tokenized_corpus)
        corpus_features = corpus_features / corpus_features.norm(dim=1, keepdim=True)
        corpus_features = corpus_features.cpu().numpy()

    if len(image_features) == 0:
        features = corpus_features
    elif len(corpus_features) == 0:
        features = image_features
    else:
        features = np.concatenate([image_features, corpus_features])
    return features, [*images, *corpus]  # type: ignore


def get_features(
    corpus: T.List[str] = [],
    images: T.List[Path] = []
) -> T.Tuple[
    np.ndarray,
    T.List[T.Union[str, Path]]
]:
    with torch.no_grad():
        return _get_features(corpus, images)


def get_2d_representation(
    features: np.ndarray
) -> T.Tuple[
    T.List[float],
    T.List[float],
]:
    umap_features = TSNE(perplexity=10).fit_transform(features)  # type: ignore

    return (
        umap_features[:, 0].tolist(),  # type: ignore
        umap_features[:, 1].tolist(),  # type: ignore
    )


def _cosine_distance(vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    vec_norm = vec / np.linalg.norm(vec)
    matrix_norm = matrix / np.linalg.norm(matrix, axis=1)[:, np.newaxis]
    cosine_similarity = np.dot(matrix_norm, vec_norm)
    cosine_distance = 1 - cosine_similarity
    return cosine_distance


def dump_corpus_features(input_path: Path, output_path: Path):
    corpus = load_corpus(str(input_path))
    corpus, corpus_description = get_features(corpus=corpus)
    embeddings = {k: v.tolist() for k, v in zip(corpus_description, corpus)}
    with open(output_path, 'wb+') as fp:
        pickle.dump(embeddings, fp)


class Vocabulary:
    def __init__(self, dumped_corpus_path: Path):
        with open(dumped_corpus_path, 'rb') as f:
            self._corpus = pickle.load(f)

    @property
    def corpus(self):
        return np.stack(list(self._corpus.values()))

    @lru_cache(1000)
    def __call__(self, key: str) -> np.ndarray:
        key = key.lower()
        try:
            return np.array(self._corpus[key])
        except KeyError:
            pass
        self._corpus[key] = get_features(corpus=[key])[0].squeeze()
        return np.array(self._corpus[key])

    def nearest(self, embedding: np.ndarray, n: int = 10) -> str:
        nearest_index = np.argsort(_cosine_distance(embedding, self.corpus))
        nearest_index = nearest_index[:n]
        return [list(self._corpus.keys())[i] for i in nearest_index]  # type: ignore
