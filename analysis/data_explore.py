import polars as pl

# eager DataFrame
df = pl.read_parquet("data/chat_messages.parquet")

# or lazy (better for large files / query pushdown)
# ldf = pl.scan_parquet("data/chat_messages.parquet")

# ldf.fetch(5)          # returns a tiny eager DataFrame (up to 5 rows)

