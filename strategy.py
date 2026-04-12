# 6h_1w_1d_Range_Breakout_Reversal_v1
# Hypothesis: On 6h timeframe, enter long when price breaks above weekly resistance (R4) with volume > 1.5x average and price > 200-period SMA (trend filter). Enter short when price breaks below weekly support (S4) with volume > 1.5x average and price < 200-period SMA. Exit on opposite breakout or when price crosses 200 SMA. Uses weekly pivots for structure, volume for confirmation, and 200 SMA for trend filter to avoid counter-trend trades. Designed for 15-35 trades/year per symbol.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_Range_Breakout_Reversal_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === WEEKLY INDICATORS: Calculate pivot points from weekly OHLC ===
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Weekly pivot points (standard calculation)
    pivot = (high_1w + low_1w + close_1w) / 3
    range_1w = high_1w - low_1w
    
    # Weekly R4 and S4 levels (more extreme breakout levels)
    r4 = pivot + range_1w * 1.1
    s4 = pivot - range_1w * 1.1
    
    # Align weekly levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    
    # === LTF INDICATORS: 200-period SMA for trend filter ===
    close_series = pd.Series(close)
    sma_200 = close_series.rolling(window=200, min_periods=200).mean().values
    
    # Volume filter: volume > 1.5 * 20-period average volume
    volume_series = pd.Series(volume)
    vol_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):  # start after warmup for SMA200
        # Skip if weekly indicators not available
        if (np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(sma_200[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Breakout conditions with volume and trend filter
        long_breakout = (close[i] > r4_aligned[i]) and volume_filter[i] and (close[i] > sma_200[i])
        short_breakout = (close[i] < s4_aligned[i]) and volume_filter[i] and (close[i] < sma_200[i])
        
        # Exit conditions: opposite breakout or trend filter violation
        exit_long = (close[i] < s4_aligned[i]) or (close[i] < sma_200[i])
        exit_short = (close[i] > r4_aligned[i]) or (close[i] > sma_200[i])
        
        if long_breakout and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_breakout and position != -1:
            position = -1
            signals[i] = -0.25
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals