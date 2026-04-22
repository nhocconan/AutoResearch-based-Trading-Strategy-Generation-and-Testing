#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian breakout with 1-day trend and volume confirmation.
Long when price breaks above 20-period Donchian upper band and 1-day EMA34 rising with volume spike.
Short when price breaks below 20-period Donchian lower band and 1-day EMA34 falling with volume spike.
Exit when price returns to Donchian midpoint or 1-day EMA34 reverses.
Donchian breakouts capture momentum; 1-day EMA provides higher-timeframe trend filter; volume spike confirms institutional participation.
Designed for low trade frequency by requiring multiple confirmations. Works in both bull and bear markets by following the 1-day trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels: 20-period high/low
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    mid_20 = (high_20 + low_20) / 2.0
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # 34-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(mid_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above upper Donchian band and 1d EMA34 rising with volume spike
            if (close[i] > high_20[i] and ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.30
                position = 1
            # Short: Price breaks below lower Donchian band and 1d EMA34 falling with volume spike
            elif (close[i] < low_20[i] and ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.30
                position = -1
        else:
            # Exit: Price returns to Donchian midpoint or 1d EMA34 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below midpoint or 1d EMA34 turns down
                if close[i] < mid_20[i] or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above midpoint or 1d EMA34 turns up
                if close[i] > mid_20[i] or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30 if position == 1 else -0.30
    
    return signals

name = "12H_DonchianBreakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0