#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: Daily donchian breakout with weekly trend filter and volume confirmation
# Enter long when price breaks above donchian(20) high, weekly EMA(50) is rising, volume > 2x average
# Enter short when price breaks below donchian(20) low, weekly EMA(50) is falling, volume > 2x average
# Exit when price reverses to opposite donchian level or volume drops below average
# Targets 50-100 trades over 4 years with strong trend-following signals

name = "1d_donchian20_1wema_vol_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA(50) for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False).mean().values
    ema_50_slope = np.diff(ema_50, prepend=ema_50[0])
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    ema_50_slope_aligned = align_htf_to_ltf(prices, df_1w, ema_50_slope)
    
    # Donchian channels (20-period) on daily data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 2x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 2.0 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(ema_50_aligned[i]) or np.isnan(ema_50_slope_aligned[i]) or
            np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price breaks below lower donchian OR weekly trend turns down
            if close[i] < low_20[i] or ema_50_slope_aligned[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper donchian OR weekly trend turns up
            if close[i] > high_20[i] or ema_50_slope_aligned[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: donchian breakout + trend + volume
            if volume[i] > volume_threshold[i]:
                if close[i] > high_20[i] and ema_50_slope_aligned[i] > 0:
                    # Break above upper donchian with rising weekly trend
                    signals[i] = 0.25
                    position = 1
                elif close[i] < low_20[i] and ema_50_slope_aligned[i] < 0:
                    # Break below lower donchian with falling weekly trend
                    signals[i] = -0.25
                    position = -1
    
    return signals