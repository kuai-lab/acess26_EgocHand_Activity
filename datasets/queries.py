from enum import Enum, auto


class BaseQueries(Enum):
    CAMINTR = auto()
    JOINTS3D = auto()
    JOINTS2D = auto()
    IMAGE = auto()
    JOINTSABS25D=auto()
    MASK = auto()
    IMAGE_T = auto()
    ACTIONIDX = auto()
    ACTIONNAME = auto()
    VERBIDX=auto()
    OBJIDX=auto()
    CAM2LOCAL=auto()
    TASKACTIONIDX=auto()
    TASKACTIONNAME=auto()
    HANDKEYPOINTS=auto() 
    OBJLABEL=auto() 
 


class TransQueries(Enum):
    CAMINTR = auto()
    JOINTS3D = auto()
    JOINTS2D = auto()
    IMAGE = auto()
    DEPTH = auto()               # ← 추가
    THERMAL = auto()             # ← 추가
    OBJIDX=auto()
    JITTERMASK = auto()
    SIDE = auto()
    SCALE = auto()
    AFFINETRANS = auto()
    ROTMAT = auto()
    MASK = auto()
    JOINTSABS25D=auto()

def one_query_in(candidate_queries, base_queries):
    for query in candidate_queries:
        if query in base_queries:
            return True
    return False


def get_trans_queries(base_queries):
    trans_queries = []
    if BaseQueries.IMAGE in base_queries:
        trans_queries.append(TransQueries.IMAGE)
        trans_queries.append(TransQueries.AFFINETRANS)
        trans_queries.append(TransQueries.ROTMAT)
        trans_queries.append(TransQueries.JITTERMASK)
        trans_queries.append(TransQueries.OBJIDX)
    if BaseQueries.JOINTS2D in base_queries:
        trans_queries.append(TransQueries.JOINTS2D)
        trans_queries.append(TransQueries.OBJIDX)
    if BaseQueries.JOINTS3D in base_queries:
        trans_queries.append(TransQueries.JOINTS3D)
        trans_queries.append(TransQueries.OBJIDX)
    if BaseQueries.CAMINTR in base_queries:
        trans_queries.append(TransQueries.CAMINTR)
        trans_queries.append(TransQueries.OBJIDX)
    if BaseQueries.JOINTSABS25D in base_queries:
        trans_queries.append(TransQueries.JOINTSABS25D)
        trans_queries.append(TransQueries.OBJIDX)
    if BaseQueries.MASK in base_queries:
        trans_queries.append(TransQueries.MASK)
    return trans_queries