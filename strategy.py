#!/usr/bin/env python3
# 4h_1d_4h_TripleBand_Breakout_V1
# Hypothesis: On 4h timeframe, price breaks above 4h upper band (ATR-based) with volume confirmation and 1d trend filter.
# Uses 1d EMA200 for trend direction (bullish when close > EMA200) and 4h ATR(14)*2 for dynamic bands.
# Entry: long when close > upper band + volume > 1.5x 20-period average in bullish trend.
# Exit: reverse when close < lower band OR trend flips bearish.
# Designed for 15-25 trades/year by requiring triple confirmation (band break, volume, trend).
# Works in bull via breakouts; works in bear via fewer false breaks due to trend filter.

name = "4h_1d_4h_TripleBand_Breakout_V1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 4h ATR(14) for dynamic bands
    # True Range
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    
    # ATR using Wilder's smoothing
    atr = np.full_like(tr, np.nan)
    if len(tr) >= 14:
        atr[13] = np.nanmean(tr[1:15])  # simple average of first 14
        for i in range(14, len(tr)):
            atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    
    # Dynamic bands: upper = close + 2*ATR, lower = close - 2*ATR
    upper_band = close + 2 * atr
    lower_band = close - 2 * atr
    
    # Volume average for spike detection
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema200_aligned[i]) or np.isnan(upper_band[i]) or 
            np.isnan(lower_band[i]) or np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Bullish trend filter: close > 1d EMA200
            if close[i] > ema200_aligned[i]:
                # Long entry: close > upper band + volume confirmation
                if (close[i] > upper_band[i] and 
                    volume[i] > 1.5 * volume_ma[i]):
                    signals[i] = 0.25
                    position = 1
        
        elif position == 1:
            # Exit conditions: reverse or trend flip
            if close[i] < lower_band[i] or close[i] < ema200_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
    
    return signals