#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian Breakout with 1d Trend Filter and Volume Spike.
Long when price breaks above Donchian upper band (20-period) and 1d EMA50 is rising with volume spike.
Short when price breaks below Donchian lower band and 1d EMA50 is falling with volume spike.
Exit when price crosses Donchian midline (10-period average of bands) or 1d EMA50 reverses.
Donchian channels provide clear breakout levels; 1d EMA ensures alignment with higher-timeframe trend;
volume spike confirms institutional participation. Designed for low trade frequency by requiring multiple confirmations.
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
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (high_20 + low_20) / 2.0
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
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
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or 
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(donchian_mid[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Break above upper band and 1d EMA50 rising with volume spike
            if close[i] > high_20[i] and ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: Break below lower band and 1d EMA50 falling with volume spike
            elif close[i] < low_20[i] and ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses midline or 1d EMA50 reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below midline or 1d EMA50 turns down
                if close[i] < donchian_mid[i] or ema50_1d_aligned[i] < ema50_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above midline or 1d EMA50 turns up
                if close[i] > donchian_mid[i] or ema50_1d_aligned[i] > ema50_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0