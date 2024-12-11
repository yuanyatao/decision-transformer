# 用来计算归一化得分的函数
import infos
def get_normalized_score(env_name, score):
    ref_min_score = infos.REF_MIN_SCORE[env_name]
    ref_max_score = infos.REF_MAX_SCORE[env_name]
    return (score - ref_min_score) / (ref_max_score - ref_min_score)

if __name__=='__main__':
    env_name = 'hopper-expert-v2'
    print(get_normalized_score(env_name,3651.96)*100.0)