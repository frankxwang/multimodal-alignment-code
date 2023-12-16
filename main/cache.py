import os
import pickle as pkl
import hashlib
from functools import wraps
from tqdm import tqdm

filepath = "cache/"

# rename cache name to exclude characters like "/"
def clean_name(name, bad_chars={"/", " "}, replacement="-"):
    for char in bad_chars:
        name = name.replace(char, replacement)
    return name


def hash_pil_image(image):
    if not hasattr(image, "hash"):
        image.hash = hashlib.md5(image.tobytes()).hexdigest()
    return image.hash


class Cache(dict):
    def __init__(self, name):
        name = clean_name(name)
        self.path = os.path.join(filepath, name) + ".pkl"

        if os.path.exists(self.path):
            with open(self.path, 'rb') as f:
                cache = pkl.load(f)
            self.update(cache)
        else:
            os.makedirs(self.path.rsplit('/', 1)[0], exist_ok=True)

    def save(self):
        with open(self.path, 'wb') as f:
            pkl.dump(dict(self), f)


class CachedStep:
    def init_cache(self, name):
        self.cache = Cache(name)

    def save_cache(self):
        self.cache.save()

    # this is based on https://stackoverflow.com/questions/59156476/use-method-from-superclass-as-decorator-for-subclass
    # we can add name as an extra thing (for instance, different versions, etc)
    @staticmethod
    def cached_func(f, name=None):
        @wraps(f)
        def cached_wrapper(self, *args, **kwargs):
            if "use_cache" in kwargs and not kwargs["use_cache"]:
                del kwargs["use_cache"]
                return f(self, *args, **kwargs)

            cache_keys = []

            zipped_args = list(zip(*args))

            for arg in args:
                # if the type is an image
                if "PIL" in str(type(arg[0])):
                    print("Image Caching:")
                    cache_keys.append(tqdm(map(hash_pil_image, arg)))
                else:
                    cache_keys.append(arg)

            # perhaps in the future it may be good to add kwargs, but for now i'm unsure if this is needed
            # cache_keys.append([kwargs] * len(args[0]))
            if name is not None:
                cache_keys.append([name] * len(args[0]))
            
            cache_keys = list(zip(*cache_keys))

            missing_indexes = [i for i, key in enumerate(cache_keys) if key not in self.cache]

            if len(missing_indexes) > 0:
                new_args = [zipped_args[i] for i in missing_indexes]
                new_results = f(self, *zip(*new_args), **kwargs)
                self.cache.update(dict([*zip([cache_keys[i] for i in missing_indexes], new_results)]))
    
                self.save_cache()

            return [self.cache[key] for key in cache_keys]
        return cached_wrapper