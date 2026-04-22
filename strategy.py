#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian(20) breakout with 1-week trend filter and volume confirmation.
Long when price breaks above Donchian upper channel with 1-week EMA50 rising and volume spike.
Short when price breaks below Donchian lower channel with 1-week EMA50 falling and volume spike.
Exit when price returns to Donchian midpoint or 1-week EMA reverses.
Designed for low trade frequency by requiring multiple confirmations and using daily timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian Channel (20-period)
    upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid = (upper + lower) / 2
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # 50-period EMA on 1-week close for trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above upper channel with 1w EMA50 rising and volume spike
            if close[i] > upper[i] and ema50_1w_aligned[i] > ema50_1w_aligned[i-1] and vol_spike:
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below lower channel with 1w EMA50 falling and volume spike
            elif close[i] < lower[i] and ema50_1w_aligned[i] < ema50_1w_aligned[i-1] and vol_spike:
                signals[i] = -0.30
                position = -1
        else:
            # Exit: Price returns to midpoint or 1-week EMA reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Price <= midpoint or 1w EMA50 turns down
                if close[i] <= mid[i] or ema50_1w_aligned[i] < ema50_1w_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price >= midpoint or 1w EMA50 turns up
                if close[i] >= mid[i] or ema50_1w_aligned[i] > ema50_1w_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "DailyDonchian20_WeeklyEMA50_Trend_Volume"
timeframe = "1d"
leverage = 1.0