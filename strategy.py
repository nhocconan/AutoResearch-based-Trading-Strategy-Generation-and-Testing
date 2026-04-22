#!/usr/bin/env python3

"""
Hypothesis: Daily 20-day Donchian breakout with weekly trend filter and volume confirmation.
Breakouts above/below the 20-day high/low with volume spike and aligned weekly trend.
Designed for low trade frequency (7-25 trades/year) by requiring strong volume confirmation
and trend alignment, reducing whipsaws in choppy markets. Works in both bull and bear markets
by following the weekly trend direction.
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
    
    # 20-day Donchian channels
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA34 for trend direction
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    weekly_close = df_weekly['close'].values
    ema34_weekly = pd.Series(weekly_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema34_weekly)
    
    # Volume confirmation: current volume > 2.0x 20-day average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema34_weekly_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: breakout above Donchian high + weekly uptrend + volume spike
            if close[i] > donch_high[i] and ema34_weekly_aligned[i] > ema34_weekly_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below Donchian low + weekly downtrend + volume spike
            elif close[i] < donch_low[i] and ema34_weekly_aligned[i] < ema34_weekly_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to midpoint of Donchian channel
            midpoint = (donch_high[i] + donch_low[i]) / 2
            if position == 1 and close[i] < midpoint:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] > midpoint:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "Daily_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
leverage = 1.0