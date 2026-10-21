"""Offline upload, provenance, isolation and graph regression checks."""
import io

import pytest
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.ingestion.loader import DocumentError, load_document_bytes, load_documents
from src.ingestion.chunker import chunk_documents
from src.ingestion.uploads import UploadCorpus
from src.retrieval.vectorstore import VectorStore


def pdf_bytes(pages, encrypted=False):
    writer = PdfWriter()
    font = DictionaryObject({NameObject('/Type'): NameObject('/Font'),
                             NameObject('/Subtype'): NameObject('/Type1'),
                             NameObject('/BaseFont'): NameObject('/Helvetica')})
    for text in pages:
        page = writer.add_blank_page(width=612, height=792)
        if text:
            page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'): DictionaryObject({NameObject('/F1'): writer._add_object(font)})})
            stream = DecodedStreamObject()
            stream.set_data(f'BT /F1 12 Tf 72 720 Td ({text}) Tj ET'.encode('ascii'))
            page[NameObject('/Contents')] = writer._add_object(stream)
    if encrypted:
        writer.encrypt('test-password')
    output = io.BytesIO()
    writer.write(output)
    return output.getvalue()


def test_pdf_pages_survive_chunking_retrieval_and_reload(tmp_path):
    docs = load_document_bytes('Mission.pdf', pdf_bytes(['The launch uses an amber rocket.', 'The return uses a cobalt capsule.']))
    assert [d.page_number for d in docs] == [1, 2]
    chunks = chunk_documents(docs)
    assert len({c.chunk_id for c in chunks}) == len(chunks) == 2
    store = VectorStore(str(tmp_path / 'index'), 'pdf_test')
    store.index_chunks(chunks, rebuild=True)
    reloaded = VectorStore(str(tmp_path / 'index'), 'pdf_test')
    hit = reloaded.similarity_search('cobalt capsule', k=1)[0]
    assert hit['page_number'] == 2
    assert hit['source_path'] == 'Mission.pdf'


@pytest.mark.parametrize('name,data,match', [
    ('empty.txt', b'', 'empty'),
    ('empty.md', b'  ', 'usable text'),
    ('binary.txt', b'\xff', 'UTF-8'),
    ('binary.txt', b'abc\0def', 'usable text'),
    ('other.exe', b'abc', 'Unsupported'),
    ('broken.pdf', b'%PDF-1.7 broken', 'corrupt'),
])
def test_invalid_files(name, data, match):
    with pytest.raises(DocumentError, match=match):
        load_document_bytes(name, data)


def test_encrypted_pdf():
    with pytest.raises(DocumentError, match='Encrypted'):
        load_document_bytes('secret.pdf', pdf_bytes(['hello'], encrypted=True))


def test_pdf_without_text_and_partial_text():
    with pytest.raises(DocumentError, match='OCR'):
        load_document_bytes('scan.pdf', pdf_bytes(['']))
    warnings = []
    docs = load_document_bytes('partial.pdf', pdf_bytes(['', 'Cobalt capsule.']), warnings=warnings)
    assert [d.page_number for d in docs] == [2]
    assert '1' in warnings[0] and 'No OCR' in warnings[0]


def test_filename_ids_and_disk_collisions(tmp_path):
    assert load_document_bytes('same.txt', b'alpha')[0].doc_id != load_document_bytes('same.md', b'alpha')[0].doc_id
    (tmp_path / 'same.txt').write_text('alpha')
    (tmp_path / 'same.md').write_text('beta')
    (tmp_path / 'nested').mkdir()
    (tmp_path / 'nested' / 'same.txt').write_text('gamma')
    docs = load_documents(tmp_path)
    assert len({d.doc_id for d in docs}) == 3


def test_session_corpora_do_not_mix_and_tiny_input_works():
    first = UploadCorpus([('Alpha.txt', b'alpha')])
    second = UploadCorpus([('Beta.txt', b'beta')])
    assert first.filenames == ['Alpha.txt']
    assert {c['source_path'] for c in first.similarity_search('alpha')} == {'Alpha.txt'}
    assert {c['source_path'] for c in second.similarity_search('beta')} == {'Beta.txt'}


def test_multiple_files_and_preamble():
    corpus = UploadCorpus([('a.md', b'Preamble about rocket.\n# Launch\nAmber propulsion.'),
                           ('b.txt', b'Cobalt capsule returns.')])
    assert corpus.file_count == 2
    assert any('Preamble' in chunk['text'] for chunk in corpus.chunks)


def test_upload_limits_and_duplicate_names(monkeypatch):
    with pytest.raises(DocumentError, match='Duplicate'):
        UploadCorpus([('a.txt', b'alpha'), ('A.txt', b'beta')])
    with pytest.raises(DocumentError, match='between'):
        UploadCorpus([])
    monkeypatch.setattr('src.ingestion.uploads.MAX_FILE_BYTES', 2)
    with pytest.raises(DocumentError, match='per-file'):
        UploadCorpus([('a.txt', b'alpha')])
    monkeypatch.setattr('src.ingestion.uploads.MAX_TOTAL_BYTES', 2)
    with pytest.raises(DocumentError, match='total'):
        UploadCorpus([('a.txt', b'alpha')])


def test_empty_rebuild_and_refit_do_not_replace_index(tmp_path):
    store = VectorStore(str(tmp_path / 'safe'), 'safe_test')
    chunks = chunk_documents(load_document_bytes('a.txt', b'Amber rocket launch.'))
    store.index_chunks(chunks, rebuild=True)
    with pytest.raises(ValueError, match='empty'):
        store.index_chunks([], rebuild=True)
    with pytest.raises(ValueError, match='rebuild'):
        store.index_chunks(chunks)
    with pytest.raises(ValueError):
        store.index_chunks(chunk_documents(load_document_bytes('b.txt', b'the and or')), rebuild=True)
    assert store.count() == 1
    assert store.similarity_search('rocket', k=1)[0]['source_path'] == 'a.txt'


def test_generation_prompt_contains_filename_and_page(monkeypatch):
    import src.generation.generator as generator
    captured = {}
    def complete(prompt, **kwargs):
        captured['prompt'] = prompt
        return 'The capsule is cobalt [1].'
    monkeypatch.setattr(generator, 'complete', complete)
    generator.generate_answer('Which capsule?', [{'source_path': 'Mission.pdf', 'page_number': 2, 'text': 'Cobalt capsule.'}])
    assert 'Mission.pdf, page: 2' in captured['prompt']


def test_upload_graph_bypasses_sample_router_and_preserves_citations(monkeypatch):
    import src.agent.nodes as nodes
    from src.agent.graph import run_agent
    from src.graders.relevance_grader import GradedChunk
    from src.graders.generation_grader import GenerationVerdict
    corpus = UploadCorpus([('Mission.pdf', pdf_bytes(['Cobalt capsule returns.']))])
    def forbidden(*args, **kwargs):
        pytest.fail('Shared sample router/store or web search was used')
    monkeypatch.setattr(nodes, 'route_question', forbidden)
    monkeypatch.setattr(nodes, 'get_vector_store', forbidden)
    monkeypatch.setattr(nodes, 'web_search', forbidden)
    monkeypatch.setattr('src.graders.relevance_grader.grade_chunk', lambda q, c: GradedChunk(c['chunk_id'], c['text'], c, True, 'mock'))
    monkeypatch.setattr(nodes, 'generate_answer', lambda q, chunks: f"Cobalt [1] from page {chunks[0]['page_number']}")
    monkeypatch.setattr(nodes, 'grade_generation', lambda *args: GenerationVerdict(True, True, [], 'mock'))
    state = run_agent('What capsule returns?', store=corpus, uploaded_corpus=True)
    assert state['route'] == 'retrieve'
    assert state['relevant_chunks'][0]['page_number'] == 1
    assert state['answer'] == 'Cobalt [1] from page 1'


def test_missing_upload_information_never_uses_web(monkeypatch):
    import src.agent.nodes as nodes
    from src.agent.graph import run_agent
    from src.graders.relevance_grader import GradedChunk
    corpus = UploadCorpus([('a.txt', b'Rocket launch uses amber propulsion.')])
    monkeypatch.setattr('src.graders.relevance_grader.grade_chunk', lambda q, c: GradedChunk(c['chunk_id'], c['text'], c, False, 'mock'))
    monkeypatch.setattr(nodes, 'rewrite_query', lambda *args: 'missing fact')
    monkeypatch.setattr(nodes, 'web_search', lambda *args: pytest.fail('Web fallback must be disabled'))
    monkeypatch.setattr(nodes, 'generate_answer', lambda *args: pytest.fail('Empty context must not invoke generation'))
    monkeypatch.setattr(nodes, 'grade_generation', lambda *args: pytest.fail('Abstention must not be graded'))
    state = run_agent('What is the ocean depth?', store=corpus, uploaded_corpus=True)
    assert not state['used_web_fallback']
    assert "don't have relevant context" in state['answer']
    assert state['abstained']
    assert state['generation_retry_count'] == 0
    assert 'generation_verdict' not in state
    assert 'Web search is disabled for uploads' in state['answer']
    assert '[Note:' not in state['answer']
