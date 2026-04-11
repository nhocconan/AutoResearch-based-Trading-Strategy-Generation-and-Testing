#!/usr/bin/env python3
"""
12h_1d_Camarilla_Pivot_Breakout_Trend
Hypothesis: Combines daily Camarilla pivot levels with 12-hour price action and volume confirmation.
Trades breakouts from key pivot levels (H3/L3) only when aligned with daily trend.
Designed for 15-30 trades/year per symbol with high win rate by avoiding false breakouts.
Works in bull/bear by following daily trend direction - avoids counter-trend losses during reversals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Camarilla_Pivot_Breakout_Trend"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for pivot levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 12-period moving average for volume filter
    vol_ma_12 = pd.Series(volume).rolling(window=12, min_periods=12).mean().values
    
    # Calculate daily Camarilla pivot levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point and ranges
    pivot = (high_1d + low_1d + close_1d) / 3
    range_hl = high_1d - low_1d
    
    # Camarilla levels: H3/L3 are key breakout levels
    h3 = close_1d + (range_hl * 1.1 / 2)
    l3 = close_1d - (range_hl * 1.1 / 2)
    
    # Daily trend filter: price above/below 20-period EMA
    ema_20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    uptrend_1d = close_1d > ema_20_1d
    downtrend_1d = close_1d < ema_20_1d
    
    # Align 1d indicators to 12h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    uptrend_1d_aligned = align_htf_to_ltf(prices, df_1d, uptrend_1d.astype(float))
    downtrend_1d_aligned = align_htf_to_ltf(prices, df_1d, downtrend_1d.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(12, n):
        # Skip if any required data is invalid
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(uptrend_1d_aligned[i]) or np.isnan(downtrend_1d_aligned[i]) or
            np.isnan(vol_ma_12[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current volume > 1.3x 12-period average
        volume_filter = volume[i] > 1.3 * vol_ma_12[i]
        
        # Breakout conditions using Camarilla H3/L3 levels
        breakout_up = close[i] > h3_aligned[i-1]  # Break above H3
        breakdown_down = close[i] < l3_aligned[i-1]  # Break below L3
        
        # Entry conditions: only trade in direction of daily trend
        long_entry = breakout_up and volume_filter and uptrend_1d_aligned[i]
        short_entry = breakdown_down and volume_filter and downtrend_1d_aligned[i]
        
        # Exit conditions: return to opposite Camarilla level or trend reversal
        long_exit = (close[i] < l3_aligned[i]) or (not uptrend_1d_aligned[i])
        short_exit = (close[i] > h3_aligned[i]) or (not downtrend_1d_aligned[i])
        
        # Priority: entry > exit > hold
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals