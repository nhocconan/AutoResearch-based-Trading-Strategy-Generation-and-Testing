#!/usr/bin/env python3
"""
Hypothesis: 12-hour Weekly Trend with Daily Volume Filter.
Long when 12h price > 1-week EMA34 and 1-day volume > 20-period average volume.
Short when 12h price < 1-week EMA34 and 1-day volume > 20-period average volume.
Exit when price crosses 1-week EMA34.
Weekly trend provides directional bias, daily volume ensures institutional participation.
Works in bull markets (follow trend) and bear markets (counter-trend bounces during low volume).
Target: 12-30 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 34:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1-week data for EMA34 - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate EMA34 on weekly close
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Load 1-day data for volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    volume_1d = df_1d['volume'].values
    avg_vol_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    avg_vol_20_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(34, n):
        # Skip if data not ready
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(avg_vol_20_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above weekly EMA34 and daily volume above average
            if close[i] > ema_34_1w_aligned[i] and volume[i] > avg_vol_20_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price below weekly EMA34 and daily volume above average
            elif close[i] < ema_34_1w_aligned[i] and volume[i] > avg_vol_20_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price falls below weekly EMA34
                if close[i] < ema_34_1w_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price rises above weekly EMA34
                if close[i] > ema_34_1w_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WeeklyEMA34_1dVolume_Filter"
timeframe = "12h"
leverage = 1.0