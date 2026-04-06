#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA filter and volume confirmation
# Enter long when: price breaks above Donchian(20) high AND price > 1d EMA(50) AND volume > 1.5x average
# Enter short when: price breaks below Donchian(20) low AND price < 1d EMA(50) AND volume > 1.5x average
# Exit when: price returns to Donchian midpoint or opposite breakout occurs
# Target: 75-200 trades over 4 years by using strict breakout conditions with trend filter

name = "4h_donchian20_1d_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) on 4h
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_roll + low_roll) / 2
    
    # 1d EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume confirmation: volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_threshold = 1.5 * volume_ma
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(high_roll[i]) or np.isnan(low_roll[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(volume_threshold[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # long position
            # Exit: price below Donchian midpoint OR opposite breakout
            if close[i] < donchian_mid[i] or close[i] < low_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above Donchian midpoint OR opposite breakout
            if close[i] > donchian_mid[i] or close[i] > high_roll[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakout entries with trend and volume confirmation
            if volume[i] > volume_threshold[i]:
                # Long breakout: price above Donchian high AND above 1d EMA
                if close[i] > high_roll[i] and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Short breakout: price below Donchian low AND below 1d EMA
                elif close[i] < low_roll[i] and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
    
    return signals