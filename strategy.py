#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian high (20) AND price > 1d EMA34 AND volume > 1.5x 20-period average.
# Short when price breaks below Donchian low (20) AND price < 1d EMA34 AND volume > 1.5x 20-period average.
# Exit when price returns to Donchian midline (10-period average of high/low) or volume drops below average.
# Designed for 12h timeframe with low trade frequency (target: 15-30/year) to avoid fee drag.
# Uses 1d EMA34 for trend filter to avoid counter-trend trades in strong trends.
# Volume filter ensures participation and avoids low-conviction moves.
name = "12h_Donchian20_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period)
    high_max20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    # Donchian midline (10-period average of high/low)
    high_avg10 = pd.Series(high).rolling(window=10, min_periods=10).mean().values
    low_avg10 = pd.Series(low).rolling(window=10, min_periods=10).mean().values
    donchian_mid = (high_avg10 + low_avg10) / 2
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 34  # Sufficient warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(high_max20[i]) or np.isnan(low_min20[i]) or np.isnan(donchian_mid[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Donchian high, price > 1d EMA34, volume filter
            long_cond = (close[i] > high_max20[i]) and (close[i] > ema34_1d_aligned[i]) and volume_filter[i]
            # Short conditions: price breaks below Donchian low, price < 1d EMA34, volume filter
            short_cond = (close[i] < low_min20[i]) and (close[i] < ema34_1d_aligned[i]) and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price returns to Donchian midline OR volume filter fails
            if (close[i] <= donchian_mid[i]) or (not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price returns to Donchian midline OR volume filter fails
            if (close[i] >= donchian_mid[i]) or (not volume_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals