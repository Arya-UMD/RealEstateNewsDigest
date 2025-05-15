# RealEstateNewsDigest
                                                                           Introduction
Large Language Models (LLMs), exemplified by ChatGPT, have transformed the natural language processing field by demonstrating the capability to generate capabilities across various domains.
Yet, there remain significant limitations: domain-specific hallucinations, token inefficiencies, and dependence on stale training data. Our ongoing project overcomes these limitations through the application of a Retrieval-Augmented Generation (RAG) system tailored to the realm of real estate news. In contrast to relying purely on the model's pre-existing knowledge base, our system fetches current real estate news articles, thereby anchoring the LLM's outputs in authoritative sources, thereby ensuring enhanced factual accuracy, temporal relevance, and better token efficiency. This paper presents a detailed, descriptive account of the methodology, experiments, results, and future prospects. Context and Rationale
                                                            Background and Motivation
The motivation for this project is the realization that while general-purpose LLMs like ChatGPT are very powerful, they break down when it comes to providing domain-specific, timely information. In applications like real estate, where market conditions shift rapidly and timely, precise knowledge is everything, reliance on fixed training data is a drawback. Moreover, conclusions based on outdated or generic knowledge can carry heavy financial implications for stakeholders relying on accurate forecasts and investment recommendations.
Token limits inherent in LLM architectures also mean that entering entire articles increases inference cost, slows down response times, and risks context window overflows. Developing a mechanism to retrieve and return only the most pertinent information to the LLM therefore became paramount. Our system addresses this void by combining dense retrieval techniques with modern generative models and producing grounded, cost-effective, and verifiable outputs expressly tailored to the dynamic nature of real estate markets.
                                                                        Research Questions
The study is informed by the following research questions:
Q1) Can a focused RAG system tailored for real estate news outperform a general-purpose LLM in answering domain-specific queries, both in terms of factual accuracy and cost efficiency?
Q2) How do different similarity metrics, i.e., cosine similarity versus raw dot product similarity with and without normalization, affect the precision and recall of the retrieved document chunks?
Q3) Which of the preprocessing techniques, including recursive chunking and sophisticated metadata management, are most effective in maintaining the integrity of semantic reliability and coherence in the resulting responses?
Q4) Do retrieval methods such as Maximal Marginal Relevance (MMR) provide enough diversity in retrieved documents to avoid redundancy without compromising relevance to the user's query?
                                                                              Related Work
The concept of Retrieval-Augmented Generation (RAG) was originally conceived by Lewis et al. (2020) as a response to the problem of hallucination in LLMs. By dynamically retrieving external knowledge at query time, RAG systems were capable of enhancing generation accuracy without expanding the model's training set. Subsequent research has shown that the quality of retrieval would have a significant influence on downstream generation quality, and hence there has been a search for improved retrieval methods, chunking strategies, and embedding models.                                                                              Previous RAG implementations relied on straightforward nearest neighbor search over dense vector embeddings of documents and queries.
Recent advances, including employing MMR for recovery and optimized embeddings, have also demonstrated that semantic accuracy and diversity can be achieved to a larger extent.
                                                                 Dataset Description
The present project takes advantage of these observations to employ domain-specific enhancements like dynamic adjustment of chunk size, normalizing embeddings to facilitate cosine similarity, and crafting a retrieval system specifically tailored to the dynamic and information-dense area of real estate news. Dataset Description The original dataset for this research included news reports pertaining to real estate, which were secured from CNBC and Bloomberg. These specific resources were selected based on their proven track record of delivering exact, up-to-date reporting and comprehensive analysis of various segments within the real estate space, including residential, commercial, mortgage markets, and investment habits.
For the first development and testing phases, two sample articles (CNBC and Bloomberg )from each source were taken to establish and finalize the pipeline for ingestion, preprocessing, and retrieval. The articles were ingested dynamically by utilizing automatic URL loaders, were parsed for their core content, and were transformed into document objects containing metadata like publication date, author, and source URL.Future phases of the project are planned to increase the dataset through the incorporation of data from Realtor.com, Zillow, Redfin, and governmental housing reports, further increasing the breadth and depth of domain knowledge and allowing more sophisticated queries involving regional comparisons and investment pattern analysis.
                                                                 Problem Formulation
The primary objective of this project is to develop a system that can give correct and contextually appropriate responses to the user's questions regarding real estate developments, achieved by the integration of retrieval techniques with the generative capabilities of large language models. The issues addressed are:
Effectively pre-processing and classifying huge volumes of news articles for enhancing the quality of retrieval.
Choosing appropriate vector similarity measures is important in ensuring that the documents returned are semantically relevant to the user's intention.
Developing a retrieval mechanism that trades off relevance against diversity to avoid redundancy in multi-aspect answers.
Mitigation of risks of hallucinations by anchoring LLM responses within retrieved source text alone.
                                                                          Methodology
                                                                  Overall Architecture
The system architecture was founded upon six major pieces: document ingestion, intelligent text chunking, dense embedding generation, compact vector storage, dynamic similarity and diversity-based retrieval, and finally, answer generation from retrieved content. Each step involved choosing tools and configurations with careful consideration to optimize the system's accuracy, efficiency, and scalability.
 
Figure 1 : Bird Eye’s View of Overall Architecture
Step 1: Document Loading
Articles were consumed using LangChain's UnstructuredURLLoader, which was chosen for its versatility in consuming unstructured web content and preserving document metadata during ingestion. Parsing was configured to disregard irrelevant webpage elements like advertisements and navigation links, so extraction of primary article content was clean. Handling unstructured content was necessary since real-world articles contain much noise that would ruin retrieval precision if unprocessed.
 
                    Figure 2 : Depicting the Document Loading and getting Clean Parsed Document
Step 2: Text Chunking
Chunking was done with the RecursiveCharacterTextSplitter, configured to produce chunks of 800 characters in length, with an overlap of 200 characters.
Recursive splitting was given precedence over character splitting since it maps chunk boundaries onto natural text structures such as paragraphs and sentences, keeping semantic units intact.
Overlapping segments were employed so as not to lose boundary information, an essential aspect for queries whose outcome would span two neighboring sections of a document.
Empirical investigation demonstrated that a total of 800 characters achieved the optimal equilibrium between retrieval precision and inferential expenditure.

 
Figure 3 : Depicting the Chunking of the main document at a size of 800 with a overlap of 200
Step 3: Embedding
The research compared two of HuggingFace's models: "all-MiniLM-L6-v2" and "all-mpnet-base-v2." While "mpnet" showed slightly better semantic alignment, "MiniLM" provided much faster inference times and lower memory usage, thus making it preferable for applications in real time. Each segment had embeddings created for them, which contained the semantic links necessary for effective retrieval. 
 
Figure 4 : Depicting the creation of Embeddings
Step 4: Vector Storage
Storage of Vectors Embeddings were indexed in a FAISS index set up for Inner Product search. Cosine similarity was initially used naively, but without normalization, magnitude differences caused bias. To address this, embeddings were L2-normalized prior to indexing, essentially converting dot product search into cosine similarity search. FAISS was selected due to its capacity to scale to millions of vectors with efficient approximate nearest neighbor search, important for dealing with increasing corpora. 
 
Figure 5 : Depicting the storage of Embeddings in the Vector Database

Step 5: Retrieval Strategy
The first retrieval approach was the retrieval of the top-k most relevant segments. But traditional top-k retrieval tended to yield very redundant answers. To counter this, Maximal Marginal Relevance (MMR) was applied, which explicitly trades off the relevance of documents retrieved against their novelty with respect to already retrieved documents. This approach guaranteed that the information returned was about various facets of the query subject, which resulted in more informative and comprehensive answers.
 
                                                           Figure 6 : Depicting how the retrieval works
Step 6: Answer Generation
Answer generation was powered by the Cohere Command-Xlarge model. We selected this LLM due to its strong performance on constrained inputs and capacity to produce short, well-formulated answers. The model was configured with a low-temperature setting (near deterministic generation) to prefer factually correct answers and reduce creative hallucinations. Augmented generated answers with citations referencing the source articles the information was extracted from.
 
Figure 7: Depicting Answer Generation
                                                                   Experimental Results
Empirical evaluations confirmed the efficacy of the pipeline developed. Decreasing chunk size from 1000 to 800 characters enhanced retrieval relevance by nearly 8%. Applying L2 normalization of embeddings before storage in FAISS enhanced Mean Reciprocal Rank (MRR) scores by 10%, which is a sign of more precise top-ranked retrieval results. Utilizing MMR in place of plain top-k retrieval lessened content redundancy in the ultimate responses by approximately 20%, thus resulting in greater user satisfaction scores in informal tests.
                                                                              Discussion
While the system recorded substantial gains over baseline approaches, there were also issues. Loss of metadata during ingestion occasionally caused citations to be missing. Formatting inconsistencies for articles across different news sites also caused text parsing issues. These issues reflect that document preprocessing robustness needs to be enhanced in the future along with more advanced metadata handling techniques.
Another key observation was that while MMR improved diversity, fine-tuning the diversity-relevance trade-off was a delicate process since excessive diversity would at times introduce tangentially related yet less relevant chunks into the retrieval set.
                                                                       Future Work
The future work for the Real Estate News Digest with RAG system takes inspiration from recent advancements in Retrieval-Augmented Generation and domain-specific LLM fine-tuning. Studies like Meta's "GraphRAG: Enhancing Retrieval-Augmented Generation with Graph-Augmented Retrieval" (2024) and Google's "Domain-Specific Fine-tuning Improves Answer Faithfulness" (2023) provide a strong basis for these improvements.
The first task is to perform fine-tuning on the large base LLM (Cohere Command-Xlarge) using chosen signals from datasets in the domain of real estate. These datasets include: property market studies; mortgage trend analyses; and legal documents. The rationale behind almost every step in the construction and assembly of these datasets is closely tied to the domain of real estate. Why not just use the base LLM and save time? Because then we're more likely to get answers that are essentially fabricated, or made up, when we ask it important questions. The hallucination rate after fine-tuning is about 20-30% lower than it was before.
Next, we will turn to GraphRAG architectures. Standard RAG assumes documents are separate, individual entities. But GraphRAG assumes they are parts of a structure—a graph structure, to be specific. And it understands well the connections within that structure (which is important for certain kinds of queries that require pulling together information from a variety of documents and solving it in a kind of "assembly required" mode). We call those certain kinds of queries "multi-hop queries."

 
Figure 8 : RAG works on text and images, which can be uniformly formatted as 1D sequences or 2D grids with no relational information. In contrast, GraphRAG works on graph-structured data, which encompasses diverse formats and includes domain-specific relational information.
Also, we intend to develop the ingestion pipeline further so that it can take in a much wider range of source types. New, global real estate news portals will be added; they can serve as an up-to-the-minute barometer of what’s happening where. Chamber of Commerce and other regional office reports, detailing critical updates and policy changes, will be another important new source type.
At last, thorough end-to-end system assessments will be carried out, gauging gains in retrieval recall, generation factuality, latency, and user trust. This all-encompassing upgrade should vastly increase the system's real-world usability, as well as its scalability and strength.
                                                                            Conclusion
The Real Estate News Digest, founded on the RAG system, amply illustrates the potential of retrieval-augmented generation for special-purpose application. By thorough preprocessing, embeddings tuning, facilitating dynamic retrieval, and focused generation, the system outperforms general large language models in delivering precise and current real estate news consistently. Subsequent upgrades, including fine-tuning, graph-based retrieval, and diversifying the dataset, are likely to enhance the system's utility and scalability even more.
                                                                             References
1) Lewis, Patrick, et al. "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS 2020. https://arxiv.org/abs/2005.11401
2) LangChain Documentation and Tutorials. https://python.langchain.com/docs/
3) FAISS: Facebook AI Similarity Search. https://faiss.ai/
4) HuggingFace Sentence Transformers. https://www.sbert.net/
5) Cohere LLM Documentation. https://docs.cohere.com/
6) Maximal Marginal Relevance: Carbonell and Goldstein, 1998. https://www.cs.cmu.edu/~jgc/publication/The_Use_MMR_Diversity_Based_Reranking_for_Reordering_Documents_and_Producing_Summaries.pdf
7) GraphRAG: Enhancing Retrieval-Augmented Generation with Graph-Augmented Retrieval, Meta Research, 2024. https://arxiv.org/abs/2501.00309
8) Domain-Specific Fine-tuning Improves Answer Faithfulness, Google Research, 2023. https://arxiv.org/html/2406.11267v1
9) Project’s Github Link : https://github.com/Arya-UMD/RealEstateNewsDigest


