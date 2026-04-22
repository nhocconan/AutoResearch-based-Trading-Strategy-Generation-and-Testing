#!/usr/bin/env python3
"""
Hypothesis: 6-hour Elder Ray Index with 1-day trend filter and volume confirmation.
Long when Bull Power > 0, Bear Power < 0, and 1-day close > EMA50 with volume above average.
Short when Bear Power < 0, Bull Power < 0, and 1-day close < EMA50 with volume above average.
Exit when Elder Ray signals reverse or volume condition fails.
Elder Ray measures bull/bear power via EMA13; 1-day trend filter ensures alignment with higher timeframe trend.
Volume filter ensures institutional participation, reducing false signals in choppy markets.
Works in both bull and bear markets by following institutional volume and higher timeframe trend.
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
    volume = prices['volume'].values
    
    # Load 1-day data for trend and volume filters - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1-day EMA50 for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 1-day average volume for volume filter
    avg_vol_1d = pd.Series(volume_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1-day indicators to 6h timeframe
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    avg_vol_1d_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, close_1d)
    
    # Calculate Elder Ray Index (13-period EMA)
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13  # Bull Power = High - EMA13
    bear_power = low - ema13   # Bear Power = Low - EMA13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):
        # Skip if data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(avg_vol_1d_aligned[i]) or np.isnan(close_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 1-day volume above average
        volume_filter = volume_1d[i] > avg_vol_1d_aligned[i]
        
        if position == 0:
            # Long: Bull Power > 0, Bear Power < 0, 1-day close > EMA50, volume above average
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                close_1d_aligned[i] > ema50_1d_aligned[i] and volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0, Bull Power < 0, 1-day close < EMA50, volume above average
            elif (bear_power[i] < 0 and bull_power[i] < 0 and 
                  close_1d_aligned[i] < ema50_1d_aligned[i] and volume_filter):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Elder Ray turns bearish OR volume filter fails
                if not (bull_power[i] > 0 and bear_power[i] < 0 and 
                        close_1d_aligned[i] > ema50_1d_aligned[i] and volume_filter):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Elder Ray turns bullish OR volume filter fails
                if not (bear_power[i] < 0 and bull_power[i] < 0 and 
                        close_1d_aligned[i] < ema50_1d_aligned[i] and volume_filter):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_1dTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0