#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout + Volume + ADX Trend Filter v1
Hypothesis: Weekly Donchian breakouts capture major trends with fewer trades. 
Daily volume confirms institutional interest. ADX filter ensures we only trade strong trends.
Works in bull/bear by using volatility-based breakouts and trend strength filter.
Target: 15-25 trades/year on 1d timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_weekly_donchian_breakout_volume_adx_v1"
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
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    highest_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    highest_high_aligned = align_htf_to_ltf(prices, df_1w, highest_high)
    lowest_low_aligned = align_htf_to_ltf(prices, df_1w, lowest_low)
    
    # Daily ADX(14) for trend strength
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    dm_plus = np.where((high - np.roll(high, 1)) > (np.roll(low, 1) - low), np.maximum(high - np.roll(high, 1), 0), 0)
    dm_minus = np.where((np.roll(low, 1) - low) > (high - np.roll(high, 1)), np.maximum(np.roll(low, 1) - low, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    tr_ma = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    dm_plus_ma = pd.Series(dm_plus).rolling(window=14, min_periods=14).mean().values
    dm_minus_ma = pd.Series(dm_minus).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * dm_plus_ma / tr_ma
    minus_di = 100 * dm_minus_ma / tr_ma
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Daily volume filter (>1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(highest_high_aligned[i]) or np.isnan(lowest_low_aligned[i]) or 
            np.isnan(adx[i]) or np.isnan(vol_filter[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price closes below weekly Donchian low OR trend weakens
            if close[i] <= lowest_low_aligned[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly Donchian high OR trend weakens
            if close[i] >= highest_high_aligned[i] or adx[i] < 20:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long breakout with strong trend and volume
            if (close[i] > highest_high_aligned[i-1] and 
                adx[i] > 25 and 
                vol_filter[i]):
                position = 1
                signals[i] = 0.25
            # Short breakdown with strong trend and volume
            elif (close[i] < lowest_low_aligned[i-1] and 
                  adx[i] > 25 and 
                  vol_filter[i]):
                position = -1
                signals[i] = -0.25
    
    return signals