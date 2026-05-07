#52417
#!/usr/bin/env python3
name = "4h_KAMA_Trend_Filter_12hVWAP_Support"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # KAMA (Kaufman Adaptive Moving Average) on 4h close
    # More responsive in trending markets, less whipsaw in ranging
    close_series = pd.Series(close)
    change = abs(close_series.diff(10))
    volatility = abs(close_series.diff(1)).rolling(window=10).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    kama = np.zeros(n)
    kama[0] = close[0]
    for i in range(1, n):
        kama[i] = kama[i-1] + sc.iloc[i] * (close[i] - kama[i-1])
    
    # 12h VWAP as dynamic support/resistance
    typical_price_12h = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    vwap_12h = (typical_price_12h * df_12h['volume']).cumsum() / df_12h['volume'].cumsum()
    vwap_12h_aligned = align_htf_to_ltf(prices, df_12h, vwap_12h.values)
    
    # Volume confirmation: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Wait for volume MA
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(vwap_12h_aligned[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA AND above 12h VWAP with volume confirmation
            price_above_kama = close[i] > kama[i]
            price_above_vwap = close[i] > vwap_12h_aligned[i]
            vol_condition = volume[i] > vol_ma_20[i] * 1.5
            
            if price_above_kama and price_above_vwap and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA AND below 12h VWAP with volume confirmation
            elif close[i] < kama[i] and close[i] < vwap_12h_aligned[i] and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below KAMA or volume drops significantly
            if close[i] < kama[i] or volume[i] < vol_ma_20[i] * 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above KAMA or volume drops significantly
            if close[i] > kama[i] or volume[i] < vol_ma_20[i] * 0.7:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h KAMA trend filter with 12h VWAP support/resistance and volume confirmation
# - KAMA adapts to market conditions: fast in trends, slow in ranges to reduce whipsaw
# - 12h VWAP provides institutional reference points for support/resistance
# - Volume confirmation (1.5x average) ensures institutional participation
# - Works in bull markets: buy when price above both KAMA and VWAP with volume
# - Works in bear markets: sell when price below both KAMA and VWAP with volume
# - Exit when price crosses KAMA (trend change) or volume drops (lack of conviction)
# - Position size 0.25 balances return potential with risk management
# - Target: 25-40 trades/year to minimize fee drag while capturing major moves