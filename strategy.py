#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian channel breakout with weekly trend filter (1-week EMA50) and volume confirmation.
# Long when price breaks above 12h Donchian upper band (20-period high) AND price > 1-week EMA50 (uptrend) AND volume > 1.5x 20-period average.
# Short when price breaks below 12h Donchian lower band (20-period low) AND price < 1-week EMA50 (downtrend) AND volume > 1.5x 20-period average.
# Exit when price crosses back below/above the Donchian mid-band (10-period average of high/low) OR volume drops below average.
# Designed for 12h timeframe with low trade frequency (target: 12-37/year) to avoid fee drag.
# Uses 1-week EMA50 for trend filter to avoid counter-trend trades and improve performance in both bull and bear markets.
# Volume filter ensures participation and avoids low-conviction moves.
name = "12h_Donchian_20_1wEMA50_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12h Donchian channel (20-period)
    high_roll = pd.Series(high)
    low_roll = pd.Series(low)
    donchian_high = high_roll.rolling(window=20, min_periods=20).max().values
    donchian_low = low_roll.rolling(window=20, min_periods=20).min().values
    donchian_mid = ((donchian_high + donchian_low) / 2).values  # Mid-band for exit
    
    # 1-week EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(ema50_1w_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, price > 1w EMA50, volume filter
            long_cond = (close[i] > donchian_high[i]) and (close[i] > ema50_1w_aligned[i]) and volume_filter[i]
            # Short conditions: price breaks below Donchian low, price < 1w EMA50, volume filter
            short_cond = (close[i] < donchian_low[i]) and (close[i] < ema50_1w_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below Donchian mid OR volume filter fails
            if close[i] < donchian_mid[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above Donchian mid OR volume filter fails
            if close[i] > donchian_mid[i] or not volume_filter[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals