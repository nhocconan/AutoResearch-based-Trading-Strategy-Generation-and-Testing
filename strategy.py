#!/usr/bin/env python3
# 6h_ThreeBarReversal_1dTrend_Volume
# Hypothesis: Uses 3-bar reversal pattern on 6h chart for mean reversion entries, filtered by 1-day EMA34 trend and volume spikes.
# In bull markets: 3-bar down reversal above EMA34 + volume spike = long.
# In bear markets: 3-bar up reversal below EMA34 + volume spike = short.
# The 3-bar reversal captures exhaustion moves, while EMA34 filter ensures trend alignment and volume confirms conviction.
# Target: 15-35 trades/year to minimize fee drag while capturing meaningful reversals.

name = "6h_ThreeBarReversal_1dTrend_Volume"
timeframe = "6h"
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
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA34 to 6h timeframe
    ema_34_1d_6h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike filter on 6h (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(35, n):  # Start after EMA warmup
        # Skip if any critical value is NaN
        if (np.isnan(ema_34_1d_6h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Three-bar down reversal: lower low and lower close for 3 consecutive bars
            three_bar_down = (low[i] < low[i-1] and low[i-1] < low[i-2] and
                              close[i] < close[i-1] and close[i-1] < close[i-2])
            # Three-bar up reversal: higher high and higher close for 3 consecutive bars
            three_bar_up = (high[i] > high[i-1] and high[i-1] > high[i-2] and
                            close[i] > close[i-1] and close[i-1] > close[i-2])
            
            # Long: 3-bar down reversal, above 1d EMA34 trend, volume spike
            if three_bar_down and close[i] > ema_34_1d_6h[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: 3-bar up reversal, below 1d EMA34 trend, volume spike
            elif three_bar_up and close[i] < ema_34_1d_6h[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price closes below EMA34 or three-bar up reversal forms
            three_bar_up = (high[i] > high[i-1] and high[i-1] > high[i-2] and
                            close[i] > close[i-1] and close[i-1] > close[i-2])
            if close[i] < ema_34_1d_6h[i] or three_bar_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price closes above EMA34 or three-bar down reversal forms
            three_bar_down = (low[i] < low[i-1] and low[i-1] < low[i-2] and
                              close[i] < close[i-1] and close[i-1] < close[i-2])
            if close[i] > ema_34_1d_6h[i] or three_bar_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals