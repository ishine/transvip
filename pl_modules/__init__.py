def get_module(cfg):
    module_cfg = cfg["module_cfg"]
    name = module_cfg["name"]
    module_cfg.pop("name")
    if name == "m4t_valle":
        from pl_modules.s2st import M4tValleModule
        return M4tValleModule(cfg, **module_cfg)
    if name == "nar":
        from pl_modules.nar import NarModule
        return NarModule(cfg, **module_cfg)
    if name == "sascodec":
        from pl_modules.sascodec import SASCModule
        return SASCModule(cfg, **module_cfg)
    raise NotImplementedError



def get_module_class(cfg):
    module_cfg = cfg["module_cfg"]
    name = module_cfg["name"]
    module_cfg.pop("name")
    if name == "m4t_valle":
        from pl_modules.s2st import M4tValleModule
        return M4tValleModule
    if name == "nar":
        from pl_modules.nar import NarModule
        return NarModule
    if name == "sasc":
        from pl_modules.sascodec import SASCModule
        return SASCModule
    raise NotImplementedError

