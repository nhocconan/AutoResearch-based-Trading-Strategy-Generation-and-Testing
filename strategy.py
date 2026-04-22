#!/usr/bin/env python3
"""
Hypothesis: 4-hour Williams Alligator with 1-day Elder Ray and volume spike.
Long when green line > red line (bullish alignment) + Bull Power > 0 + volume spike.
Short when red line > green line (bearish alignment) + Bear Power < 0 + volume spike.
Exit when alignment reverses or volume drops below average.
Williams Alligator identifies trend, Elder Ray measures bull/bear power, volume confirms.
Designed for low trade frequency by requiring trend alignment and power confirmation.
Works in both bull and bear markets by following the daily trend via Elder Ray.
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
    
    # Load 1-day data for Elder Ray (EMA13) - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray to 4h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Williams Alligator on 4h data (Smoothed SMAs)
    # Jaw: 13-period SMMA, Teeth: 8-period SMMA, Lips: 5-period SMMA
    # Using SMA as approximation for SMMA with sufficient lookback
    jaw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    teeth = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    lips = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(13, n):  # Start after enough data for Alligator jaw
        # Skip if data not ready
        if (np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Bullish alignment (Lips > Teeth > Jaw) + Bull Power > 0 + volume spike
            if (lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                bull_power_1d_aligned[i] > 0 and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bearish alignment (Jaw > Teeth > Lips) + Bear Power < 0 + volume spike
            elif (jaw[i] > teeth[i] and teeth[i] > lips[i] and 
                  bear_power_1d_aligned[i] < 0 and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Alignment reverses OR volume drops below average
            exit_signal = False
            
            if position == 1:
                # Exit long: Bearish alignment OR Bull Power <= 0
                if (jaw[i] > teeth[i] or teeth[i] > lips[i] or 
                    bull_power_1d_aligned[i] <= 0):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bullish alignment OR Bear Power >= 0
                if (lips[i] > teeth[i] or teeth[i] > jaw[i] or 
                    bear_power_1d_aligned[i] >= 0):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_WilliamsAlligator_ElderRay_Volume"
timeframe = "4h"
leverage = 1.0