import pandas as pd
import numpy as np
from ta import add_all_ta_features

# 1. Load 1-minute BTC data
df = pd.read_csv("btcusd_1-min_data.csv")
df['datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
df.set_index('datetime', inplace=True)

# 2. Resample to 15-minute OHLCV
df_15min = df.resample('15T').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

# 3. Filter from 2020 onwards
df_15min = df_15min[df_15min.index >= '2020-01-01']

# 4. Add technical indicators using `ta`
df_ta = add_all_ta_features(
    df_15min.copy(), open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
)

# 5. Add binary indicator features
df_ta['rsi_7'] = df_ta['momentum_rsi']
df_ta['rsi_14'] = df_ta['momentum_rsi'].rolling(14).mean()
df_ta['rsi_golden_cross'] = ((df_ta['rsi_7'] > df_ta['rsi_14']) &
                             (df_ta['rsi_7'].shift(1) <= df_ta['rsi_14'].shift(1))).astype(int)

df_ta['macd_cross_up'] = ((df_ta['trend_macd'] > df_ta['trend_macd_signal']) &
                          (df_ta['trend_macd'].shift(1) <= df_ta['trend_macd_signal'].shift(1))).astype(int)

df_ta['bb_contraction'] = (df_ta['volatility_bbw'] < 0.1).astype(int)
df_ta['adx_strong'] = (df_ta['trend_adx'] > 25).astype(int)
df_ta['bb_break_upper'] = (df_ta['Close'] > df_ta['volatility_bbh']).astype(int)

# 6. Add label based on future return
sequence_len = 60
horizon = 5
alpha = 0.01

df_ta['label'] = np.nan
for i in range(len(df_ta) - sequence_len - horizon):
    current = df_ta['Close'].iloc[i + sequence_len - 1]
    future = df_ta['Close'].iloc[i + sequence_len - 1 + horizon]
    ret = (future - current) / current

    if ret > alpha:
        df_ta.iloc[i + sequence_len - 1, df_ta.columns.get_loc('label')] = 1
    elif ret < -alpha:
        df_ta.iloc[i + sequence_len - 1, df_ta.columns.get_loc('label')] = 0

# 7. Export to CSV
columns_to_export = [
    'Open', 'High', 'Low', 'Close', 'Volume',
    'rsi_golden_cross', 'macd_cross_up', 'bb_contraction',
    'adx_strong', 'bb_break_upper', 'label'
]

df_export = df_ta[columns_to_export]
df_export.reset_index().to_csv("btc_15min_with_indicators_and_labels.csv", index=False)
print("✅ Saved to btc_15min_with_indicators_and_labels.csv")