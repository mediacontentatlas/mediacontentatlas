## Anonymous Submission for CHI'25 👋


As digital media use continues to evolve and influence various aspects of life, developing flexible and scalable tools to study complex media experiences is essential. This study introduces the Media Content Atlas (MCA), a novel pipeline designed to help researchers investigate large-scale screen data beyond traditional screen-use metrics. Leveraging state-of-the-art multimodal large language models (MLLMs), MCA enables moment-by-moment content analysis, content-based clustering, topic modeling, image retrieval, and interactive visualizations. Evaluated on 1.12 million smartphone screenshots continuously captured during screen use from 112 adults over an entire month, MCA facilitates open-ended exploration and hypothesis generation as well as hypothesis-driven investigations at an unprecedented scale. Expert evaluators underscored its usability and potential for research and intervention design, with clustering results rated 96% relevant and descriptions 83% accurate. By bridging methodological possibilities with domain-specific needs, MCA accelerates both inductive and deductive inquiry, presenting new opportunities for media and HCI research.

![image](https://github.com/mediacontentatlas/mediacontentatlas/blob/main/assets/mcapipeline.png)

In this repo, you will find the code files for the following parts of MCA: 
## MCA Pipeline
- [Generate Screenshot Image Embeddings with CLIP](https://github.com/mediacontentatlas/mediacontentatlas/blob/main/mca_pipeline/anonymized_clip_embedding_generation.py) 
- [Generate Screenshot Descriptions with Llava-OneVision](https://github.com/mediacontentatlas/mediacontentatlas/blob/main/mca_pipeline/anonymized_description_generation.py)
- [Generate Description Embeddings using GTE-large](https://github.com/mediacontentatlas/mediacontentatlas/blob/main/mca_pipeline/anonymized_description_embedding_generation.py)
- [Conduct Clustering and Topic Modeling with BERTopic and Llama2](https://github.com/mediacontentatlas/mediacontentatlas/blob/main/mca_pipeline/anonymized_clustering_topicmodeling_example.py)
- [Create Interactive Visualizations with DataMapPlot-and Customize it!](https://github.com/mediacontentatlas/mediacontentatlas/blob/main/mca_pipeline/anonymized_create_interactive_visualizations.ipynb)
- [Retrieve Relevant Images with CLIP and LlaVa-OneVision+GTE-Large](https://github.com/mediacontentatlas/mediacontentatlas/blob/main/mca_pipeline/anonymized_image_retrieval_app.py)
## MCA Evaluation
- [Expert Survey 1](https://github.com/mediacontentatlas/mediacontentatlas/blob/main/expert_surveys/anonymized_survey1.py) 
- [Expert Survey 2](https://github.com/mediacontentatlas/mediacontentatlas/blob/main/expert_surveys/anonymized_survey2.py) 
- [Expert Survey 3](https://github.com/mediacontentatlas/mediacontentatlas/blob/main/expert_surveys/anonymized_survey3.py)  

## Notes
- This repo will be updated regularly for better reproducibility, meanwhile open issues for any questions.
- Working on creating a synthetic dataset to showcase the whole pipeline.

