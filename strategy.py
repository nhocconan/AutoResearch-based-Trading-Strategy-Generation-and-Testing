#!/usr/bin/env python3
"""
1d Donchian Breakout with Weekly Trend Filter and Volume Confirmation
Hypothesis: Donchian breakouts capture strong momentum, weekly trend filter ensures
alignment with higher timeframe direction, and volume confirmation filters false breakouts.
This combination works in both bull and bear markets by catching sustained moves while
avoiding chop. Designed for low trade frequency (target: 15-25 trades/year) to minimize
fee drag.
"""

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
    
    # Get weekly data for trend filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Weekly EMA20 for trend filter
    ema20_1w = pd.Series(df_1w['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Daily Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Warmup for indicators
    
    for i in range(start_idx, n):
        if np.isnan(ema20_1w_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        weekly_trend = ema20_1w_aligned[i]
        upper_channel = highest_high[i]
        lower_channel = lowest_low[i]
        vol_ok = vol_confirm[i]
        
        if position == 0:
            # Enter long: price breaks above upper channel, weekly uptrend, volume confirmation
            if close[i] > upper_channel and close[i] > weekly_trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below lower channel, weekly downtrend, volume confirmation
            elif close[i] < lower_channel and close[i] < weekly_trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns to or below weekly EMA (trend change)
            if close[i] <= weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns to or above weekly EMA (trend change)
            if close[i] >= weekly_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0