#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R3/S3 breakout with 12h trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 AND 12h EMA50 above EMA200 AND volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S3 AND 12h EMA50 below EMA200 AND volume > 1.5x 20-period average.
# Exit when price crosses back inside the Camarilla H-L range (between S1 and R1).
# Camarilla provides reversal structure, 12h EMA trend filters higher timeframe bias, volume confirms institutional participation.
# Target: 75-200 total trades over 4 years (19-50/year).

name = "4h_Camarilla_R3S3_Breakout_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla calculation (from previous day's OHLC)
    df_d = get_htf_data(prices, '1d')
    if len(df_d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla levels
    prev_close = df_d['close'].shift(1).values
    prev_high = df_d['high'].shift(1).values
    prev_low = df_d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    R3 = prev_close + (prev_range * 1.1000 / 4)
    S3 = prev_close - (prev_range * 1.1000 / 4)
    R1 = prev_close + (prev_range * 1.1000 / 12)
    S1 = prev_close - (prev_range * 1.1000 / 12)
    
    # Align Camarilla levels to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_d, S3)
    R1_aligned = align_htf_to_ltf(prices, df_d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_d, S1)
    
    # 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 200:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema200_12h = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align EMAs to 4h timeframe
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema200_12h_aligned = align_htf_to_ltf(prices, df_12h, ema200_12h)
    
    # Trend filter: EMA50 above/below EMA200
    trend_up = ema50_12h_aligned > ema200_12h_aligned
    trend_down = ema50_12h_aligned < ema200_12h_aligned
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 1)  # Sufficient warmup for EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(trend_up[i]) or np.isnan(trend_down[i]) or
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3, trend up, volume filter
            long_cond = (close[i] > R3_aligned[i]) and trend_up[i] and volume_filter[i]
            # Short conditions: price breaks below Camarilla S3, trend down, volume filter
            short_cond = (close[i] < S3_aligned[i]) and trend_down[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back below Camarilla S1
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back above Camarilla R1
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals