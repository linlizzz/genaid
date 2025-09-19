from preview_inputs import main_preview_inputs
from build_faiss_index import main_build_faiss_index
from experiment_retrieval import main_experiment_retrieval
import sys 
sys.path.append("/scratch/work/zhangl9/genaid/") 
from export_tables_plots import main_export_tables_plots

print("Preview inputs start...")
main_preview_inputs()
print("Preview inputs done!")

print("Build faiss index start...")
# main_build_faiss_index(embed_model=["intfloat/multilingual-e5-large-instruct"])
print("Build faiss index done!")

print("Experiment retrieval start...")
# modify retriever_mode ["dense", "hybrid", "hybrid_rerank"]
# main_experiment_retrieval(query_model="intfloat/multilingual-e5-large-instruct", retriever_mode="hybrid_rerank")
print("Experiment retrieval done!")

print("Export tables plots start...")
# modify retriever_mode ["dense", "hybrid", "hybrid_rerank"]
# main_export_tables_plots(embed_model_dir_name="intfloat_multilingual-e5-large-instruct", retriever_mode="hybrid_rerank")
print("Export tables plots done!")