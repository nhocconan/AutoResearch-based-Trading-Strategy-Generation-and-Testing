#!/usr/bin/env python3
"""
Hypothesis: 6-hour Elder Ray Index combined with 1-day trend filter and volume confirmation.
Long when Bull Power > 0 (close > EMA13) and Bear Power < 0 (low < EMA13) with volume > 1.5x 20-period average.
Short when Bear Power > 0 (low > EMA13) and Bull Power < 0 (close < EMA13) with volume > 1.5x 20-period average.
Exit when Elder Ray signals reverse or volume drops below average.
Uses 1-day EMA50 for trend filter: only take longs when price > EMA50, shorts when price < EMA50.
Designed for low trade frequency (~15-25/year) to avoid fee drag while capturing strong institutional moves.
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
    
    # Load 1-day data for EMA50 trend filter and EMA13 for Elder Ray - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA13 and EMA50 on 1-day data
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Elder Ray components: Bull Power = Close - EMA13, Bear Power = Low - EMA13
    bull_power_1d = close_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align indicators to 6h timeframe (wait for prior day's close)
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Volume filter: 20-period average volume on 6h timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(ema13_aligned[i]) or np.isnan(ema50_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0, Bear Power < 0, price > EMA50, volume confirmation
            if (bull_power_aligned[i] > 0 and bear_power_aligned[i] < 0 and 
                close[i] > ema50_aligned[i] and volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power > 0, Bull Power < 0, price < EMA50, volume confirmation
            elif (bear_power_aligned[i] > 0 and bull_power_aligned[i] < 0 and 
                  close[i] < ema50_aligned[i] and volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Reverse Elder Ray signal or volume drop
            exit_signal = False
            
            if position == 1:
                # Exit long: Bear Power becomes positive OR Bull Power negative OR volume drops
                if (bear_power_aligned[i] >= 0 or bull_power_aligned[i] <= 0 or 
                    volume[i] < vol_ma[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bull Power becomes positive OR Bear Power negative OR volume drops
                if (bull_power_aligned[i] >= 0 or bear_power_aligned[i] <= 0 or 
                    volume[i] < vol_ma[i]):
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