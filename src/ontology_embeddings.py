def load_ontology_embeddings(emb_path="concept_embeddings.json"):
    import json
    with open(emb_path, "r") as f:
        concept_dict = json.load(f)
    return concept_dict

def get_ontology_vector_for_patch(patch_info, concept_dict):
    # patch_info might contain { 'is_burned':True, 'imperv_ratio':0.5, ... }
    # logic to pick relevant concepts
    relevant_concepts = []
    if patch_info.get("is_burned", False):
        relevant_concepts.append("BurnArea")
    if patch_info.get("imperv_ratio",0)<=0.4:
        relevant_concepts.append("ImperviousSurface")
    # etc...
    # sum or average their embeddings
    # in practice, project them to 768
    pass
