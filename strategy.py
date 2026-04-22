#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian channel breakout with 1-day volume confirmation and 1-week trend filter.
Long when price breaks above Donchian(20) high with volume > 1.5x 20-period average and 1-week EMA50 rising.
Short when price breaks below Donchian(20) low with volume > 1.5x 20-period average and 1-week EMA50 falling.
Exit when price crosses Donchian midline (10-period average) or 1-week EMA50 reverses.
Donchian channels provide clear breakout levels; volume confirms institutional interest; weekly EMA filters for higher-timeframe trend.
Designed for low trade frequency by requiring multiple confirmations and using wide channels.
Works in both bull and bear markets by following the weekly trend direction.
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
    
    # Donchian channels: 20-period high/low
    def rolling_max(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            res[i] = np.max(arr[i-window+1:i+1])
        return res
    
    def rolling_min(arr, window):
        res = np.full_like(arr, np.nan, dtype=float)
        for i in range(window-1, len(arr)):
            res[i] = np.min(arr[i-window+1:i+1])
        return res
    
    donchian_high = rolling_max(high, 20)
    donchian_low = rolling_min(low, 20)
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Load 1w data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 1w close for trend
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = np.convolve(volume, np.ones(20)/20, mode='same')
    vol_ma_20[:19] = np.nan
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high with volume spike and 1w EMA50 rising
            if close[i] > donchian_high[i] and vol_spike and ema50_1w_aligned[i] > ema50_1w_aligned[i-1]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with volume spike and 1w EMA50 falling
            elif close[i] < donchian_low[i] and vol_spike and ema50_1w_aligned[i] < ema50_1w_aligned[i-1]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses Donchian midline or 1w EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below midline or 1w EMA50 turns down
                if close[i] < donchian_mid[i] or ema50_1w_aligned[i] < ema50_1w_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above midline or 1w EMA50 turns up
                if close[i] > donchian_mid[i] or ema50_1w_aligned[i] > ema50_1w_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_DonchianBreakout_1wTrend_Volume"
timeframe = "4h"
leverage = 1.0