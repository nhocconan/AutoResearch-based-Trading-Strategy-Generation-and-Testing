#!/usr/bin/env python3
# 4H_1D_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Use Camarilla pivot levels from daily timeframe (R1/S1) for breakout entries with daily trend filter (EMA34) and volume confirmation (1.5x average volume). Works in bull/bear by following daily trend. Target: 20-30 trades/year per symbol (80-120 total over 4 years).

name = "4H_1D_Camarilla_R1_S1_Breakout_1dTrend_Volume"
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
    
    # Get 1d data for Camarilla pivots, trend, and volume average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for each day
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # R1 = C + (H - L) * 1.1 / 12
    # S1 = C - (H - L) * 1.1 / 12
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    r1_1d = close_1d + range_1d * 1.1 / 12.0
    s1_1d = close_1d - range_1d * 1.1 / 12.0
    
    # Daily EMA34 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Daily average volume for volume confirmation
    vol_1d_series = pd.Series(df_1d['volume'].values)
    avg_vol_1d = vol_1d_series.rolling(window=20, min_periods=1).mean().values
    
    # Align all 1d indicators to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    avg_vol_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(avg_vol_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x daily average volume
        vol_confirm = volume[i] > (avg_vol_aligned[i] * 1.5)
        
        if position == 0:
            # Enter long: price breaks above R1 + daily uptrend (close > EMA34) + volume confirmation
            if (close[i] > r1_aligned[i]) and (close_1d[-1] > ema34_aligned[i]) and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below S1 + daily downtrend (close < EMA34) + volume confirmation
            elif (close[i] < s1_aligned[i]) and (close_1d[-1] < ema34_aligned[i]) and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below S1 or daily trend turns bearish
            if (close[i] < s1_aligned[i]) or (close_1d[-1] < ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above R1 or daily trend turns bullish
            if (close[i] > r1_aligned[i]) or (close_1d[-1] > ema34_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals