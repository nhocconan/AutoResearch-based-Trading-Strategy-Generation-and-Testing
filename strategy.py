#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian breakout with 1-day trend filter and volume confirmation.
Long when price breaks above 20-period Donchian upper band on 12h and 1d EMA50 rising with volume spike.
Short when price breaks below 20-period Donchian lower band on 12h and 1d EMA50 falling with volume spike.
Exit when price crosses 12-period EMA on 12h or trend reverses.
Designed for low trade frequency by requiring multiple confirmations and using higher timeframes.
Works in both bull and bear markets by following the 1d trend.
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
    
    # Load 12h data for price channel and trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # 20-period Donchian channels on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # 12-period EMA on 12h for exit
    close_12h = df_12h['close'].values
    ema12_12h = pd.Series(close_12h).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema12_12h_aligned = align_htf_to_ltf(prices, df_12h, ema12_12h)
    
    # Load 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 50-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema12_12h_aligned[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above Donchian high and 1d EMA50 rising with volume spike
            if (close[i] > donchian_high_aligned[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Break below Donchian low and 1d EMA50 falling with volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses 12-period EMA on 12h or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Price below EMA12 or 1d EMA50 turns down
                if close[i] < ema12_12h_aligned[i] or ema50_1d_aligned[i] < ema50_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price above EMA12 or 1d EMA50 turns up
                if close[i] > ema12_12h_aligned[i] or ema50_1d_aligned[i] > ema50_1d_aligned[i-1]:
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