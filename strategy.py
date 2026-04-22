#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout with 12-hour EMA trend filter and volume confirmation.
Long when price breaks above 20-period Donchian high with 12h EMA50 rising and volume spike.
Short when price breaks below 20-period Donchian low with 12h EMA50 falling and volume spike.
Exit when price crosses the 20-period Donchian mean or 12h EMA50 reverses.
Donchian channels identify volatility breakouts; 12h EMA provides higher-timeframe trend filter;
volume spike confirms institutional participation. Designed for low trade frequency by requiring multiple confirmations.
Works in both bull and bear markets by following the 12h trend.
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
    
    # Donchian channel: 20-period high and low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mean = (donchian_high + donchian_low) / 2
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 50-period EMA on 12h close for trend
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(donchian_mean[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high with 12h EMA50 rising and volume spike
            if (close[i] > donchian_high[i-1] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low with 12h EMA50 falling and volume spike
            elif (close[i] < donchian_low[i-1] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses Donchian mean or 12h EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below Donchian mean or 12h EMA50 turns down
                if close[i] < donchian_mean[i] or ema50_12h_aligned[i] < ema50_12h_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above Donchian mean or 12h EMA50 turns up
                if close[i] > donchian_mean[i] or ema50_12h_aligned[i] > ema50_12h_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_DonchianBreakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0