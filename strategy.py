#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h trend filter and 4h volume confirmation.
# Long when price breaks above Donchian upper(20) AND 12h EMA50 > 12h EMA100 (uptrend) AND 4h volume > 1.5x 30-period average.
# Short when price breaks below Donchian lower(20) AND 12h EMA50 < 12h EMA100 (downtrend) AND 4h volume > 1.5x 30-period average.
# Exit when price crosses the opposite Donchian band (lower for long exit, upper for short exit).
# Focus on strong breakouts with trend and volume to reduce false signals and trade frequency.
# Target: 60-120 total trades over 4 years (15-30/year) for low fee drift.

name = "4h_Donchian_20_12hTrend_4hVolume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 100:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA50 and EMA100 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema100_12h = pd.Series(close_12h).ewm(span=100, adjust=False, min_periods=100).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    ema100_12h_aligned = align_htf_to_ltf(prices, df_12h, ema100_12h)
    
    # 4h volume filter: current volume > 1.5x 30-period average
    vol_ma30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    volume_filter = volume > (1.5 * vol_ma30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Sufficient warmup for Donchian and EMAs
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(ema100_12h_aligned[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: break above upper band, uptrend (EMA50 > EMA100), volume spike
            long_cond = (close[i] > high_roll[i]) and (ema50_12h_aligned[i] > ema100_12h_aligned[i]) and volume_filter[i]
            # Short conditions: break below lower band, downtrend (EMA50 < EMA100), volume spike
            short_cond = (close[i] < low_roll[i]) and (ema50_12h_aligned[i] < ema100_12h_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below lower band (breakdown)
            if close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above upper band (breakout)
            if close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals