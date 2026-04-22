#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams Alligator with 1-day Trend Filter and Volume Confirmation.
Long when price is above Alligator's Jaw during 1-day uptrend with volume spike.
Short when price is below Alligator's Jaw during 1-day downtrend with volume spike.
Exit when price crosses the Alligator's Teeth or trend reverses.
Williams Alligator uses smoothed moving averages (SMMA) of median price.
Designed for low trade frequency by requiring trend alignment and volume confirmation.
Works in both bull and bear markets by following the 1-day trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - same as Wilder's smoothing"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: SMMA = (prev_SMMA * (period-1) + current_price) / period
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Median price for Alligator
    median_price = (high + low) / 2
    
    # Williams Alligator: SMMA(13,8), SMMA(8,5), SMMA(5,3) of median price
    jaw = smma(median_price, 13)  # Blue line (slowest)
    teeth = smma(median_price, 8)  # Red line (middle)
    lips = smma(median_price, 5)   # Green line (fastest)
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 20-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema20_1d = pd.Series(close_1d).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema20_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Increased warmup for Alligator
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema20_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price above Jaw (Alligator bullish alignment) + 1d uptrend + volume spike
            # Bullish alignment: Lips > Teeth > Jaw
            if (close[i] > jaw[i] and lips[i] > teeth[i] and teeth[i] > jaw[i] and 
                ema20_1d_aligned[i] > ema20_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: price below Jaw (Alligator bearish alignment) + 1d downtrend + volume spike
            # Bearish alignment: Lips < Teeth < Jaw
            elif (close[i] < jaw[i] and lips[i] < teeth[i] and teeth[i] < jaw[i] and 
                  ema20_1d_aligned[i] < ema20_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses Teeth or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price below Teeth or 1d trend turns down
                if close[i] < teeth[i] or ema20_1d_aligned[i] < ema20_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above Teeth or 1d trend turns up
                if close[i] > teeth[i] or ema20_1d_aligned[i] > ema20_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Williams_Alligator_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0