from pandera import Check, Column, DataFrameSchema, Index, SeriesSchema

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
        "price": Column(int, required=False),
        "x": Column(float),
        "y": Column(float),
        "z": Column(float),
    },
    index=Index(int),
    strict=True,
    coerce=True,
)

PREPROCESSED_SCHEMA = DataFrameSchema(
    columns={
        "carat": Column(float),
        "cut": Column(str, checks=Check.isin(CUT)),
        "color": Column(str, checks=Check.isin(COLOR)),
        "clarity": Column(str, checks=Check.isin(CLARITY)),
        "depth": Column(float),
        "table": Column(float),
        "price": Column(int, required=False),
        "x": Column(float),
        "y": Column(float),
        "z": Column(float),
        "density": Column(float),
        "x-y": Column(float),
        "y-z": Column(float),
        "z-x": Column(float),
        "x/y": Column(float),
        "y/z": Column(float),
        "z/x": Column(float),
        "x-median_x": Column(float),
        "y-median_y": Column(float),
        "z-median_z": Column(float),
        "median_carat_by_cut": Column(float),
        "carat-median_carat_by_cut": Column(float),
        "carat/median_carat_by_cut": Column(float),
        "median_carat_by_color": Column(float),
        "carat-median_carat_by_color": Column(float),
        "carat/median_carat_by_color": Column(float),
        "median_carat_by_clarity": Column(float),
        "carat-median_carat_by_clarity": Column(float),
        "carat/median_carat_by_clarity": Column(float),
        "rate_cut*color": Column(float),
        "rate_color*clarity": Column(float),
        "rate_clarity*cut": Column(float),
    },
    index=Index(int),
    strict=True,
    coerce=True,
)

X_SCHEMA = DataFrameSchema(
    columns={
        "carat": Column(float),
        "cut": Column(str, checks=Check.isin(CUT)),
        "color": Column(str, checks=Check.isin(COLOR)),
        "clarity": Column(str, checks=Check.isin(CLARITY)),
        "depth": Column(float),
        "table": Column(float),
        "x": Column(float),
        "y": Column(float),
        "z": Column(float),
        "density": Column(float),
        "x-y": Column(float),
        "y-z": Column(float),
        "z-x": Column(float),
        "x/y": Column(float),
        "y/z": Column(float),
        "z/x": Column(float),
        "x-median_x": Column(float),
        "y-median_y": Column(float),
        "z-median_z": Column(float),
        "median_carat_by_cut": Column(float),
        "carat-median_carat_by_cut": Column(float),
        "carat/median_carat_by_cut": Column(float),
        "median_carat_by_color": Column(float),
        "carat-median_carat_by_color": Column(float),
        "carat/median_carat_by_color": Column(float),
        "median_carat_by_clarity": Column(float),
        "carat-median_carat_by_clarity": Column(float),
        "carat/median_carat_by_clarity": Column(float),
        "rate_cut*color": Column(float),
        "rate_color*clarity": Column(float),
        "rate_clarity*cut": Column(float),
    },
    index=Index(int),
    strict=True,
    coerce=True,
)

Y_SCHEMA = SeriesSchema(
    int,
    nullable=False,
    coerce=True,
)

RAW_PREDICTION_SCHEMA = DataFrameSchema(
    columns={
        "carat": Column(float),
        "cut": Column(str, checks=Check.isin(CUT)),
        "color": Column(str, checks=Check.isin(COLOR)),
        "clarity": Column(str, checks=Check.isin(CLARITY)),
        "depth": Column(float),
        "table": Column(float),
        "x": Column(float),
        "y": Column(float),
        "z": Column(float),
    },
    index=Index(int),
    strict=True,
    coerce=True,
)
