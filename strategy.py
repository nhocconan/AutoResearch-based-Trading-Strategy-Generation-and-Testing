#!/usr/bin/env python3
# Hypothesis: 1d Donchian breakout with 1w trend filter and volume spike
# Long when price breaks above Donchian upper band (20-period) with 1w EMA50 uptrend and volume > 2x average
# Short when price breaks below Donchian lower band (20-period) with 1w EMA50 downtrend and volume > 2x average
# Exit when price retouches Donchian middle (mean of upper/lower) or reverses to opposite band
# Uses Donchian for breakout structure, 1w EMA for trend filter, volume for conviction
# Designed to capture multi-day breakouts with low frequency (<25/year) to minimize fee drag
# Target: 30-80 total trades over 4 years (7-20/year) with size 0.25

name = "1d_Donchian_Breakout_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    high_rolling = pd.Series(high).rolling(window=20, min_periods=20).max()
    low_rolling = pd.Series(low).rolling(window=20, min_periods=20).min()
    upper = high_rolling.values
    lower = low_rolling.values
    middle = (upper + lower) / 2
    
    # Calculate 1w EMA50 for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    vol_confirm = volume > (2.0 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and Donchian calculation
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(middle[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above upper band, 1w EMA50 uptrend, volume spike
            if (close[i] > upper[i] and 
                ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and  # EMA rising
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower band, 1w EMA50 downtrend, volume spike
            elif (close[i] < lower[i] and 
                  ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and  # EMA falling
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price retouches middle band or reverses to lower band
            if (close[i] <= middle[i]) or (close[i] < lower[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price retouches middle band or reverses to upper band
            if (close[i] >= middle[i]) or (close[i] > upper[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals