# 4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS
# Hypothesis: Camarilla R1/S1 breakout with 1d trend filter and volume spike exploits
# institutional support/resistance levels. Works in bull (breakouts) and bear (reversals)
# due to mean-reverting nature of Camarilla levels in ranges and breakout strength
# in trends. Volume confirmation reduces false signals.

#!/usr/bin/env python3
name = "4h_Camarilla_R1_S1_Breakout_1dTrend_VolumeS"
timeframe = "4h"
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
    
    # 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # H, L, C from previous day
    H = high_1d[:-1]  # previous day high
    L = low_1d[:-1]   # previous day low
    C = close_1d[:-1] # previous day close
    
    # Camarilla calculations
    range_hl = H - L
    R1 = C + (range_hl * 1.1 / 12)
    R2 = C + (range_hl * 1.1 / 6)
    R3 = C + (range_hl * 1.1 / 4)
    S1 = C - (range_hl * 1.1 / 12)
    S2 = C - (range_hl * 1.1 / 6)
    S3 = C - (range_hl * 1.1 / 4)
    
    # Shift to align with current 4h bars (previous day's levels)
    R1 = np.concatenate([[np.nan], R1])
    R2 = np.concatenate([[np.nan], R2])
    R3 = np.concatenate([[np.nan], R3])
    S1 = np.concatenate([[np.nan], S1])
    S2 = np.concatenate([[np.nan], S2])
    S3 = np.concatenate([[np.nan], S3])
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to 4h
    R1_4h = align_htf_to_ltf(prices, df_1d, R1)
    R2_4h = align_htf_to_ltf(prices, df_1d, R2)
    R3_4h = align_htf_to_ltf(prices, df_1d, R3)
    S1_4h = align_htf_to_ltf(prices, df_1d, S1)
    S2_4h = align_htf_to_ltf(prices, df_1d, S2)
    S3_4h = align_htf_to_ltf(prices, df_1d, S3)
    ema34_1d_4h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 35  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R1_4h[i]) or np.isnan(S1_4h[i]) or 
            np.isnan(ema34_1d_4h[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with volume spike and uptrend (close > EMA34)
            if (close[i] > R1_4h[i] and vol_spike[i] and close[i] > ema34_1d_4h[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume spike and downtrend (close < EMA34)
            elif (close[i] < S1_4h[i] and vol_spike[i] and close[i] < ema34_1d_4h[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 (mean reversion) or loses trend
            if (close[i] < S1_4h[i] or close[i] < ema34_1d_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 (mean reversion) or loses trend
            if (close[i] > R1_4h[i] or close[i] > ema34_1d_4h[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals