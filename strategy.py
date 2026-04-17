#!/usr/bin/env python3
"""
12h Donchian Breakout with Daily Trend Filter
Long: Price breaks above Donchian(20) high + 1d EMA(34) rising
Short: Price breaks below Donchian(20) low + 1d EMA(34) falling
Exit: Opposite Donchian break or price crosses EMA(34)
Designed to capture trend continuations with clear breakouts.
Target: 50-150 total trades over 4 years (12-37/year) on 12h timeframe
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels on 12h
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max()
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min()
    donchian_high = period20_high.values
    donchian_low = period20_low.values
    
    # Get 1d data for trend filter (EMA34)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1d EMA(34) to 12h
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 20  # need Donchian calculations
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema_34_1d_aligned[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        ema34 = ema_34_1d_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high + EMA34 rising
            if price > upper and ema34 > ema_34_1d_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low + EMA34 falling
            elif price < lower and ema34 < ema_34_1d_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: break below Donchian low OR price crosses below EMA34
            if price < lower or price < ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: break above Donchian high OR price crosses above EMA34
            if price > upper or price > ema34:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian_Breakout_DailyTrend"
timeframe = "12h"
leverage = 1.0