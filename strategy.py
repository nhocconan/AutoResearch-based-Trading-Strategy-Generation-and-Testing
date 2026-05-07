#!/usr/bin/env python3
# 4h_WilliamsAlligator_ElderRay_Signal
# Hypothesis: Combines Williams Alligator trend (Jaw/Teeth/Lips) with Elder Ray (Bull/Bear Power) on 1d.
# Long when Green line above Red (bullish alignment) AND Bull Power > 0 with rising trend.
# Short when Red line above Green (bearish alignment) AND Bear Power < 0 with falling trend.
# Uses 1d timeframe for alignment, reducing whipsaws vs lower timeframes.
# Target: 20-40 trades/year to minimize fee drag while maintaining edge in both bull and bear markets.

name = "4h_WilliamsAlligator_ElderRay_Signal"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for Williams Alligator and Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams Alligator: Jaw (13-period SMMA, shifted 8), Teeth (8-period SMMA, shifted 5), Lips (5-period SMMA, shifted 3)
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (PREV_SMMA * (period-1) + CURRENT_VALUE) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(close_1d, 13)  # Blue line (Jaw)
    teeth = smma(close_1d, 8)  # Red line (Teeth)
    lips = smma(close_1d, 5)   # Green line (Lips)
    
    # Shift the lines as per Alligator definition
    jaw = np.roll(jaw, 8)   # Jaw shifted 8 bars forward
    teeth = np.roll(teeth, 5) # Teeth shifted 5 bars forward
    lips = np.roll(lips, 3)   # Lips shifted 3 bars forward
    
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema_13_1d
    bear_power = low_1d - ema_13_1d
    
    # Align all indicators to 4h timeframe
    jaw_4h = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_4h = align_htf_to_ltf(prices, df_1d, teeth)
    lips_4h = align_htf_to_ltf(prices, df_1d, lips)
    bull_power_4h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_4h = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Volume spike filter on 4h (20-period average)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)  # Reduced threshold to avoid too few trades
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN
        if (np.isnan(jaw_4h[i]) or np.isnan(teeth_4h[i]) or np.isnan(lips_4h[i]) or 
            np.isnan(bull_power_4h[i]) or np.isnan(bear_power_4h[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Lips > Teeth (Green above Red) AND Bull Power > 0 AND rising AND volume spike
            if lips_4h[i] > teeth_4h[i] and bull_power_4h[i] > 0 and bull_power_4h[i] > bull_power_4h[i-1] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: Teeth > Lips (Red above Green) AND Bear Power < 0 AND falling AND volume spike
            elif teeth_4h[i] > lips_4h[i] and bear_power_4h[i] < 0 and bear_power_4h[i] < bear_power_4h[i-1] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Lips <= Teeth OR Bull Power <= 0
            if lips_4h[i] <= teeth_4h[i] or bull_power_4h[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Teeth <= Lips OR Bear Power >= 0
            if teeth_4h[i] <= lips_4h[i] or bear_power_4h[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals