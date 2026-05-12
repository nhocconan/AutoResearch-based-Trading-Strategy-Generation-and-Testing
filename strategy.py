# 6h_12h_1d_Supertrend_Volume_Breakout
# Hypothesis: Uses 12h Supertrend (ATR=10, mult=3) for trend direction, and 12h volume breakout
# for entry timing. Enters long when price breaks above 12h Supertrend upper band with volume spike,
# and short when price breaks below 12h Supertrend lower band with volume spike.
# Designed for low trade frequency (~50-150 total trades over 4 years) to minimize fee drag.
# Works in bull/bear markets by following 12h trend while using 6h price action for precise entries.
# Avoids overtrading by requiring volume confirmation and using Supertrend as a dynamic trend filter.

name = "6h_12h_1d_Supertrend_Volume_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h data for Supertrend calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR(10) for 12h
    tr1 = np.maximum(high_12h[1:], low_12h[:-1]) - np.minimum(high_12h[1:], low_12h[:-1])
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Supertrend upper and lower bands
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (3 * atr)
    lower_band = hl2 - (3 * atr)
    
    # Initialize Supertrend arrays
    supertrend = np.full_like(close_12h, np.nan)
    uptrend = np.full_like(close_12h, True)
    
    # Calculate Supertrend
    for i in range(1, len(close_12h)):
        if np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            supertrend[i] = np.nan
            uptrend[i] = uptrend[i-1] if i > 0 else True
            continue
            
        if close_12h[i] > upper_band[i-1]:
            uptrend[i] = True
        elif close_12h[i] < lower_band[i-1]:
            uptrend[i] = False
        else:
            uptrend[i] = uptrend[i-1]
            if uptrend[i] and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if not uptrend[i] and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        supertrend[i] = lower_band[i] if uptrend[i] else upper_band[i]
    
    # Volume spike: >2.0x 20-period average (on 12h timeframe)
    vol_ma_12h = pd.Series(df_12h['volume'].values).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = df_12h['volume'].values > (2.0 * vol_ma_12h)
    
    # Align all indicators to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    uptrend_aligned = align_htf_to_ltf(prices, df_12h, uptrend.astype(float))  # bool to float for alignment
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike_12h.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        if (np.isnan(supertrend_aligned[i]) or
            np.isnan(uptrend_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Supertrend (uptrend) + volume spike
            if (close[i] > supertrend_aligned[i] and 
                uptrend_aligned[i] > 0.5 and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Supertrend (downtrend) + volume spike
            elif (close[i] < supertrend_aligned[i] and 
                  uptrend_aligned[i] < 0.5 and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Supertrend
            if close[i] < supertrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Supertrend
            if close[i] > supertrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals