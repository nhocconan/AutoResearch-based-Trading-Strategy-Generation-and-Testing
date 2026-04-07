#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_volume_v1
Hypothesis: On 12h timeframe, enter long when price breaks above the 1d Camarilla R3 level with above-average volume, short when price breaks below S3 level with above-average volume. Use the 1d ATR percentile to filter for low volatility regimes where breakouts are more likely to succeed. Exit when price returns to the 1d Camarilla Pivot level. Target: 50-150 total trades over 4 years (12-37/year) to balance opportunity with fee minimization. Works in both bull and bear markets by fading false breakouts in high volatility and capturing true breakouts in low volatility.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_volume_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot and Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels: H5 = close + 1.1*range/2, H4 = close + 1.1*range/4, H3 = close + 1.1*range/6
    # L3 = close - 1.1*range/6, L4 = close - 1.1*range/4, L5 = close - 1.1*range/2
    r3 = close_1d + 1.1 * range_1d / 6.0
    s3 = close_1d - 1.1 * range_1d / 6.0
    pivot_level = pivot
    
    # Align to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_level)
    
    # Calculate 1d ATR for volatility regime filter
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # True Range
    tr1 = high_1d[1:] - low_1d[1:]
    tr2 = np.abs(high_1d[1:] - close_1d[:-1])
    tr3 = np.abs(low_1d[1:] - close_1d[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # First value NaN
    
    # ATR(14)
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # ATR percentile rank (252-day lookback for 1-year)
    atr_percentile = pd.Series(atr_1d).rolling(window=252, min_periods=50).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else np.nan, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # Volume moving average for confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(atr_percentile_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(close[i])):
            signals[i] = 0.0
            continue
        
        # Low volatility regime: ATR percentile below 50th percentile
        low_vol = atr_percentile_aligned[i] < 0.5
        
        # Volume confirmation: above average volume
        vol_ok = volume[i] > vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price returns to pivot level or volatility increases significantly
            if close[i] <= pivot_aligned[i] or atr_percentile_aligned[i] > 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price returns to pivot level or volatility increases significantly
            if close[i] >= pivot_aligned[i] or atr_percentile_aligned[i] > 0.8:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            if low_vol and vol_ok:
                # Breakout above R3 with volume - go long
                if close[i] > r3_aligned[i] and close[i-1] <= r3_aligned[i-1]:
                    position = 1
                    signals[i] = 0.25
                # Breakout below S3 with volume - go short
                elif close[i] < s3_aligned[i] and close[i-1] >= s3_aligned[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals