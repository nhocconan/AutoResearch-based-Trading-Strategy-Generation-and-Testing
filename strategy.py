#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout with 1w volume spike and 1w EMA50 trend filter.
# Long when price breaks above Donchian(20) upper band AND 1w volume > 1.8x 24-period average AND price > 1w EMA50.
# Short when price breaks below Donchian(20) lower band AND 1w volume > 1.8x 24-period average AND price < 1w EMA50.
# Exit when price crosses back below/above 1w EMA50 (trend-based exit).
# Uses Donchian breakout for trend capture, volume confirmation for strength, weekly EMA for trend filter.
# Target: 40-80 total trades over 4 years (10-20/year) with controlled frequency.

name = "1d_Donchian_20_1wVolume_1wEMA50"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w data for Donchian, volume filter, and EMA trend
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period)
    high_20 = pd.Series(df_1w['high'].values).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low'].values).rolling(window=20, min_periods=20).min().values
    
    # 1w volume filter: current volume > 1.8x 24-period average
    vol_ma24 = pd.Series(df_1w['volume'].values).rolling(window=24, min_periods=24).mean().values
    volume_filter = df_1w['volume'].values > (1.8 * vol_ma24)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1w indicators to 1d timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    volume_filter_aligned = align_htf_to_ltf(prices, df_1w, volume_filter)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for EMA
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(volume_filter_aligned[i]) or np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian upper, volume spike, above 1w EMA50
            long_cond = (close[i] > high_20_aligned[i]) and volume_filter_aligned[i] and (close[i] > ema50_1w_aligned[i])
            # Short conditions: price breaks below Donchian lower, volume spike, below 1w EMA50
            short_cond = (close[i] < low_20_aligned[i]) and volume_filter_aligned[i] and (close[i] < ema50_1w_aligned[i])
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 1w EMA50 (trend change)
            if close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 1w EMA50 (trend change)
            if close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals