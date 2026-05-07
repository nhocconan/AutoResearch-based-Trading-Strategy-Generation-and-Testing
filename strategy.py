#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot point breakout with 4h trend filter and volume confirmation.
# Long when price breaks above Camarilla R3 (4h) AND 4h close > 4h EMA50 (uptrend) AND volume > 1.5x 20-period average.
# Short when price breaks below Camarilla S3 (4h) AND 4h close < 4h EMA50 (downtrend) AND volume > 1.5x 20-period average.
# Exit when price returns to Camarilla Pivot (4h) OR trend reverses (4h close crosses EMA50).
# Designed for 1h timeframe with moderate trade frequency (target: 15-35/year) to avoid fee drag.
# Uses 4h Camarilla levels for structure and 4h EMA50 for trend filter to avoid counter-trend trades.
# Volume filter ensures participation and avoids low-conviction moves.
name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 4h data for Camarilla levels and EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla levels for each 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla formula: Range = high - low
    # R3 = close + 1.1 * (high - low) / 2
    # S3 = close - 1.1 * (high - low) / 2
    # Pivot = (high + low + close) / 3
    rng_4h = high_4h - low_4h
    r3_4h = close_4h + 1.1 * rng_4h / 2.0
    s3_4h = close_4h - 1.1 * rng_4h / 2.0
    pivot_4h = (high_4h + low_4h + close_4h) / 3.0
    
    # Align Camarilla levels to 1h
    r3_4h_aligned = align_htf_to_ltf(prices, df_4h, r3_4h)
    s3_4h_aligned = align_htf_to_ltf(prices, df_4h, s3_4h)
    pivot_4h_aligned = align_htf_to_ltf(prices, df_4h, pivot_4h)
    
    # 4h EMA50 for trend filter
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(r3_4h_aligned[i]) or np.isnan(s3_4h_aligned[i]) or 
            np.isnan(pivot_4h_aligned[i]) or np.isnan(ema50_4h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3, 4h uptrend, volume filter
            long_cond = (close[i] > r3_4h_aligned[i]) and (close_4h[-1] > ema50_4h[-1] if len(close_4h) > 0 else False) and volume_filter[i]
            # Short conditions: price breaks below S3, 4h downtrend, volume filter
            short_cond = (close[i] < s3_4h_aligned[i]) and (close_4h[-1] < ema50_4h[-1] if len(close_4h) > 0 else False) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.20
                position = 1
            elif short_cond:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price returns to pivot OR 4h trend reverses
            if close[i] <= pivot_4h_aligned[i] or close_4h[-1] < ema50_4h[-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price returns to pivot OR 4h trend reverses
            if close[i] >= pivot_4h_aligned[i] or close_4h[-1] > ema50_4h[-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals