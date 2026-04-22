#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams Alligator with Elder Ray and volume filter.
Long when Alligator jaws (13) > teeth (8) > lips (5) and Elder Ray bull power > 0 with volume spike.
Short when Alligator jaws < teeth < lips and Elder Ray bear power < 0 with volume spike.
Exit when Alligator lines cross or Elder Ray power changes sign.
Williams Alligator identifies trend direction, Elder Ray measures trend strength,
volume spike confirms institutional participation. Designed for low trade frequency
by requiring multiple confirmations and using 12h timeframe. Works in both bull
and bear markets by following the trend direction with strict entry conditions.
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
    
    # Load 1-day data for Elder Ray calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # Calculate Elder Ray (Bull Power and Bear Power) on daily data
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power_1d = high_1d - ema13_1d
    bear_power_1d = low_1d - ema13_1d
    
    # Align Elder Ray to 12h timeframe
    bull_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_1d_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # Williams Alligator on 12h data directly (no HTF needed for this)
    # Jaw = 13-period SMMA (smoothed moving average) of median price
    # Teeth = 8-period SMMA of median price
    # Lips = 5-period SMMA of median price
    median_price = (high + low) / 2.0
    
    def smma(arr, period):
        """Smoothed Moving Average"""
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        result = np.full_like(arr, np.nan, dtype=float)
        # First value is simple SMA
        result[period-1] = np.mean(arr[:period])
        # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    jaw = smma(median_price, 13)
    teeth = smma(median_price, 8)
    lips = smma(median_price, 5)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for Alligator
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power_1d_aligned[i]) or np.isnan(bear_power_1d_aligned[i]) or
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        # Alligator conditions
        bullish_alligator = jaw[i] > teeth[i] > lips[i]
        bearish_alligator = jaw[i] < teeth[i] < lips[i]
        
        if position == 0:
            # Long: Bullish Alligator alignment + Elder Ray bull power > 0 + volume spike
            if (bullish_alligator and 
                bull_power_1d_aligned[i] > 0 and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + Elder Ray bear power < 0 + volume spike
            elif (bearish_alligator and 
                  bear_power_1d_aligned[i] < 0 and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Alligator turns bearish OR Elder Ray bull power <= 0
                if not bullish_alligator or bull_power_1d_aligned[i] <= 0:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Alligator turns bullish OR Elder Ray bear power >= 0
                if not bearish_alligator or bear_power_1d_aligned[i] >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_WilliamsAlligator_ElderRay_Volume"
timeframe = "12h"
leverage = 1.0