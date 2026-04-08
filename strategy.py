#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Confirmation and ATR Stop
Hypothesis: Weekly Donchian breakouts on daily timeframe capture strong trends while minimizing trades. Volume confirmation filters false breakouts. ATR-based stops manage risk. Works in bull/bear by using volatility-adjusted stops. Targets 10-25 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_v2"
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
    
    # Weekly Donchian Channel (20-period)
    df_1w = get_htf_data(prices, '1w')
    highest_high_1w = df_1w['high'].rolling(window=20, min_periods=20).max().values
    lowest_low_1w = df_1w['low'].rolling(window=20, min_periods=20).min().values
    highest_high_1w_aligned = align_htf_to_ltf(prices, df_1w, highest_high_1w)
    lowest_low_1w_aligned = align_htf_to_ltf(prices, df_1w, lowest_low_1w)
    
    # Daily ATR(14) for stop loss
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter (>1.3x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_1w_aligned[i]) or np.isnan(lowest_low_1w_aligned[i]) or 
            np.isnan(atr[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly Donchian low OR ATR stop
            if (close[i] <= lowest_low_1w_aligned[i] or
                close[i] <= (highest_high_1w_aligned[i-1] - 2.5 * atr[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly Donchian high OR ATR stop
            if (close[i] >= highest_high_1w_aligned[i] or
                close[i] >= (lowest_low_1w_aligned[i-1] + 2.5 * atr[i])):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout with volume confirmation
            if (close[i] > highest_high_1w_aligned[i-1] and vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown with volume confirmation
            elif (close[i] < lowest_low_1w_aligned[i-1] and vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals