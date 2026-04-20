#!/usr/bin/env python3
"""
4h_Camarilla_Pivot_R1S1_Breakout_Volume_EMAFilter
Hypothesis: Camarilla pivot levels (R1/S1) from 1d combined with 4h EMA50 trend filter and volume confirmation.
Long when price breaks above R1 with volume > 1.5x 20-period average and price > EMA50.
Short when price breaks below S1 with volume > 1.5x 20-period average and price < EMA50.
Uses 4h as primary timeframe for structure, 1d for pivots, and volume for confirmation.
Target: 15-25 trades per year per symbol with position size 0.25 to manage drawdown.
Works in bull markets via long breakouts and bear markets via short breakdowns.
"""

name = "4h_Camarilla_Pivot_R1S1_Breakout_Volume_EMAFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for 1d
    def calculate_camarilla(high, low, close):
        # Typical price
        typical_price = (high + low + close) / 3
        # Pivot point
        pivot = (high + low + close) / 3
        # Ranges
        range_hl = high - low
        # Camarilla levels
        r1 = pivot + (range_hl * 1.1 / 12)
        s1 = pivot - (range_hl * 1.1 / 12)
        return r1, s1
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_r1 = np.zeros(len(close_1d))
    camarilla_s1 = np.zeros(len(close_1d))
    for i in range(len(close_1d)):
        r1, s1 = calculate_camarilla(high_1d[i], low_1d[i], close_1d[i])
        camarilla_r1[i] = r1
        camarilla_s1[i] = s1
    
    # Align Camarilla levels to 4h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Calculate 4h EMA50 for trend filter
    close_series = pd.Series(close)
    ema50 = close_series.ewm(span=50, adjust=False, min_periods=50).values
    
    # Calculate volume average (20-period)
    volume_series = pd.Series(volume)
    vol_avg = volume_series.rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or 
            np.isnan(ema50[i]) or np.isnan(vol_avg[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R1, volume > 1.5x average, price > EMA50
            if (close[i] > camarilla_r1_aligned[i] and 
                volume[i] > 1.5 * vol_avg[i] and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1, volume > 1.5x average, price < EMA50
            elif (close[i] < camarilla_s1_aligned[i] and 
                  volume[i] > 1.5 * vol_avg[i] and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price breaks below S1 OR volume < 0.5x average
            if (close[i] < camarilla_s1_aligned[i] or 
                volume[i] < 0.5 * vol_avg[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price breaks above R1 OR volume < 0.5x average
            if (close[i] > camarilla_r1_aligned[i] or 
                volume[i] < 0.5 * vol_avg[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals