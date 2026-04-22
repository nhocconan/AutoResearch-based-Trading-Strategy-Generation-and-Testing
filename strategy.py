#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian breakout with 1-day trend and volume confirmation.
Long when price breaks above Donchian high(20) and 1-day EMA34 rising with volume spike.
Short when price breaks below Donchian low(20) and 1-day EMA34 falling with volume spike.
Exit when price crosses Donchian midline or 1-day EMA34 reverses.
Donchian provides clear breakout levels, 1-day EMA filters trend direction, volume confirms participation.
Designed for low trade frequency by requiring multiple confirmations. Works in both bull and bear by following 1-day trend.
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
    
    # Donchian channels: 20-period high and low
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            res[i] = np.max(arr[i - window + 1:i + 1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window - 1, len(arr)):
            res[i] = np.min(arr[i - window + 1:i + 1])
        return res
    
    donch_high = rolling_max(high, 20)
    donch_low = rolling_min(low, 20)
    donch_mid = (donch_high + donch_low) / 2.0
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 34-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or np.isnan(donch_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: break above Donchian high and 1-day EMA34 rising with volume spike
            if (close[i] > donch_high[i] and 
                ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low and 1-day EMA34 falling with volume spike
            elif (close[i] < donch_low[i] and 
                  ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses Donchian midline or 1-day EMA34 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price below midline or 1-day EMA34 turns down
                if close[i] < donch_mid[i] or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above midline or 1-day EMA34 turns up
                if close[i] > donch_mid[i] or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_DonchianBreakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0