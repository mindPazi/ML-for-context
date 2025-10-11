from evaluation.cosqa_loader import CoSQALoader


def test_load():
    loader = CoSQALoader()
    corpus, queries, relevance = loader.load(split="test")
    
    assert len(corpus) > 0
    assert len(queries) > 0
    assert len(relevance) > 0
    
    query = queries[0]
    assert "query_id" in query
    assert "query_text" in query
    assert query["query_id"] in relevance
    assert isinstance(relevance[query["query_id"]], list)
