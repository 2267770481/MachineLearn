def wrap(fun):
    def inner(*args, **kwargs):
        print(f"------------开始执行{fun.__name__}...--------------")
        res = fun(*args, **kwargs)
        print(f'------------结束执行{fun.__name__}...--------------\n')
        return res

    return inner
