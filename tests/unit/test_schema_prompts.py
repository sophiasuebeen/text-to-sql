from text_to_sql.prompts import SYSTEM_PROMPT, build_messages
from text_to_sql.schema import quote_ident, serialize_schema


def test_quote_ident_escapes_quotes():
    assert quote_ident('we"ird') == '"we""ird"'
    assert quote_ident("order") == '"order"'


def test_serialize_keys_order_and_types(make_db):
    db = make_db(
        """
        CREATE TABLE zeta (b TEXT, a INTEGER);
        CREATE TABLE "Alpha Table" ("my id" INTEGER, "order" varchar(20), x, PRIMARY KEY ("order", "my id"));
        CREATE TABLE child (cid INTEGER PRIMARY KEY, p1 INTEGER, p2 TEXT, z TEXT,
            FOREIGN KEY (p1, p2) REFERENCES "Alpha Table" ("my id", "order"),
            FOREIGN KEY (z) REFERENCES zeta);
        """
    )
    assert serialize_schema(db) == (
        'CREATE TABLE "Alpha Table" (\n'
        '  "my id" INTEGER,\n'
        '  "order" varchar(20),\n'
        '  "x",\n'
        '  PRIMARY KEY ("order", "my id")\n'
        ");\n\n"
        'CREATE TABLE "child" (\n'
        '  "cid" INTEGER,\n'
        '  "p1" INTEGER,\n'
        '  "p2" TEXT,\n'
        '  "z" TEXT,\n'
        '  PRIMARY KEY ("cid"),\n'
        '  FOREIGN KEY ("p1", "p2") REFERENCES "Alpha Table" ("my id", "order"),\n'
        '  FOREIGN KEY ("z") REFERENCES "zeta"\n'
        ");\n\n"
        'CREATE TABLE "zeta" (\n'
        '  "b" TEXT,\n'
        '  "a" INTEGER\n'
        ");"
    )


def test_serialize_is_deterministic_and_skips_internal(make_db):
    db = make_db("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, v TEXT); INSERT INTO t(v) VALUES ('a');")
    s = serialize_schema(db)
    assert s == serialize_schema(db)
    assert "sqlite_sequence" not in s


def test_messages_shape():
    m = build_messages("CREATE TABLE t (x);", "  How many? ", sql=" SELECT 1 ")
    assert [x["role"] for x in m] == ["system", "user", "assistant"]
    assert m[0]["content"] == SYSTEM_PROMPT
    assert m[1]["content"].endswith("Question: How many?")
    assert m[2]["content"] == "SELECT 1"
    assert len(build_messages("s", "q")) == 2
