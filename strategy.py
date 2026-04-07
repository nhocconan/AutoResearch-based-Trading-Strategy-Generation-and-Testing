#!/usr/bin/env python3
"""
6h_donchian_breakout_weekly_pivot_v1
Hypothesis: On 6h timeframe, combine 20-period Donchian breakout with weekly pivot direction for trend filter and volume confirmation. Enter long when price breaks above Donchian upper band with price above weekly pivot (bullish bias) and volume > 1.5x average; enter short when price breaks below Donchian lower band with price below weekly pivot (bearish bias) and volume > 1.5x average. Exit when price returns to Donchian middle (mean reversion) or opposite breakout occurs. This strategy captures strong trending moves with institutional participation (volume) while using weekly pivot for multi-timeframe alignment, reducing false signals. Works in bull/bear via pivot-based trend filter and breakout logic.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian_breakout_weekly_pivot_v1"
timeframe = "6h"
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
    
    # Weekly data for pivot calculation
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (using prior week's data)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    close_weekly = df_weekly['close'].values
    
    # Weekly pivot: P = (H + L + C) / 3
    pivot_weekly = (high_weekly + low_weekly + close_weekly) / 3
    # Support/resistance levels
    r1 = 2 * pivot_weekly - low_weekly
    s1 = 2 * pivot_weekly - high_weekly
    r2 = pivot_weekly + (high_weekly - low_weekly)
    s2 = pivot_weekly - (high_weekly - low_weekly)
    r3 = high_weekly + 2 * (pivot_weekly - low_weekly)
    s3 = low_weekly - 2 * (high_weekly - pivot_weekly)
    
    # Align weekly pivot to 6h timeframe (use prior week's pivot for no look-ahead)
    pivot_6h = align_htf_to_ltf(prices, df_weekly, pivot_weekly)
    r3_6h = align_htf_to_ltf(prices, df_weekly, r3)
    s3_6h = align_htf_to_ltf(prices, df_weekly, s3)
    
    # Donchian channels on 6h (20-period)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2
    
    # Volume confirmation (20-period average on 6h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if required data not available
        if (np.isnan(pivot_6h[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit conditions
            exit_long = False
            # Exit if price returns to Donchian middle (mean reversion)
            if close[i] <= donchian_mid[i]:
                exit_long = True
            # Exit if price breaks below Donchian lower (reversal)
            elif close[i] < lowest_low[i]:
                exit_long = True
            
            if exit_long:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit conditions
            exit_short = False
            # Exit if price returns to Donchian middle (mean reversion)
            if close[i] >= donchian_mid[i]:
                exit_short = True
            # Exit if price breaks above Donchian upper (reversal)
            elif close[i] > highest_high[i]:
                exit_short = True
            
            if exit_short:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry conditions
            long_entry = False
            # Price breaks above Donchian upper with bullish bias (above weekly pivot) and volume confirmation
            if close[i] > highest_high[i] and close[i-1] <= highest_high[i-1]:
                # Bullish bias: price above weekly pivot (or above S3 for stronger bias)
                if close[i] > pivot_6h[i] and vol_confirm:
                    long_entry = True
            
            # Short entry conditions
            short_entry = False
            # Price breaks below Donchian lower with bearish bias (below weekly pivot) and volume confirmation
            if close[i] < lowest_low[i] and close[i-1] >= lowest_low[i-1]:
                # Bearish bias: price below weekly pivot (or below S3 for stronger bias)
                if close[i] < pivot_6h[i] and vol_confirm:
                    short_entry = True
            
            if long_entry:
                position = 1
                signals[i] = 0.25
            elif short_entry:
                position = -1
                signals[i] = -0.25
    
    return signals