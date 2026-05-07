#!/usr/bin/env python3
name = "6h_Pivot_R3S3_Breakout_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 1d High, Low, Close for pivot calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each 1d bar
    # Pivot = (H + L + C) / 3
    # R3 = C + (H - L) * 1.1 / 2
    # S3 = C - (H - L) * 1.1 / 2
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r3_1d = close_1d + (high_1d - low_1d) * 1.1 / 2.0
    s3_1d = close_1d - (high_1d - low_1d) * 1.1 / 2.0
    
    # Align 1d pivot levels to 6h timeframe
    pivot_1d_aligned = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 6h volume filter: > 1.5x 24-period average (48h)
    vol_ma_6h = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    vol_filter = volume > 1.5 * vol_ma_6h
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 24)  # Wait for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(pivot_1d_aligned[i]) or np.isnan(r3_1d_aligned[i]) or 
            np.isnan(s3_1d_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(vol_ma_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Break above R3 with uptrend (price > EMA34) and volume
            if (close[i] > r3_1d_aligned[i] and close[i] > ema34_1d_aligned[i] and vol_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with downtrend (price < EMA34) and volume
            elif (close[i] < s3_1d_aligned[i] and close[i] < ema34_1d_aligned[i] and vol_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price < EMA34 (trend change) or re-enter R3 range
            if close[i] < ema34_1d_aligned[i] or close[i] < r3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price > EMA34 (trend change) or re-enter S3 range
            if close[i] > ema34_1d_aligned[i] or close[i] > s3_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Camarilla R3/S3 levels act as significant support/resistance.
# Breakouts above R3 in uptrend (price > EMA34) or below S3 in downtrend (price < EMA34)
# with volume confirmation capture institutional flow.
# Works in bull (buy R3 breakouts) and bear (sell S3 breakdowns) regimes.
# Uses 1d pivots for structure, 6s for entry, and volume to filter false breakouts.
# Target: 20-40 trades/year to minimize fee drag. Position size 0.25 limits risk.