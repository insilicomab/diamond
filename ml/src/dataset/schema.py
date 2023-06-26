from pandera import Check, Column, DataFrameSchema, Index

CUT = ["Ideal", "Premium", "Good", "Very Good", "Fair"]

COLOR = ["E", "I", "J", "H", "F", "G", "D"]

CLARITY = ["SI2", "SI1", "VS1", "VS2", "VVS2", "VVS1", "I1", "IF"]

BASE_SCHEMA = DataFrameSchema(
    columns={
        "carat": Column(float),
        "cut": Column(str, checks=Check.isin(CUT)),
        "color": Column(str, checks=Check.isin(COLOR)),
        "clarity": Column(str, checks=Check.isin(CLARITY)),
        "depth": Column(float),
        "table": Column(float),
        "price": Column(int),
        "x": Column(float),
        "y": Column(float),
        "z": Column(float),
    },
    index=Index(int),
    strict=True,
    coerce=True,
)
