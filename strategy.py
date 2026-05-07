#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_12hTrend_Volume"
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
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 12h EMA20 for trend filter
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Daily Camarilla pivot levels (using previous day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    # Previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    # Calculate Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 4
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 4
    # Align to 4h
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    
    # Volume spike detection (4h)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(R3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_condition = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 0:
            # Long: break above R3 in 12h uptrend with volume
            if close[i] > R3_aligned[i] and vol_condition and ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: break below S3 in 12h downtrend with volume
            elif close[i] < S3_aligned[i] and vol_condition and ema_20_12h_aligned[i] < ema_20_12h_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below S3 or trend change
            if close[i] < S3_aligned[i] or ema_20_12h_aligned[i] < ema_20_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above R3 or trend change
            if close[i] > R3_aligned[i] or ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h trend filter and volume confirmation
# - Camarilla R3/S3 levels act as strong support/resistance derived from prior day's range
# - Breakout above R3 in 12h uptrend signals bullish continuation
# - Breakdown below S3 in 12h downtrend signals bearish continuation
# - Volume confirmation (1.5x average) reduces false breakouts
# - 12h EMA20 trend filter ensures alignment with higher timeframe trend
# - Exit when price returns to opposite S3/R3 level or trend changes
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend)
# - Uses actual daily Camarilla calculations (not resampled) for accuracy
# - Aims for 80-200 total trades over 4 years (20-50/year) to stay within limits