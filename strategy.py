#!/usr/bin/env python3
"""
Hypothesis: 12-hour Williams Alligator with 1-week Trend Filter and Volume Confirmation.
Long when price is above Alligator's Jaw during 1-week uptrend with volume spike.
Short when price is below Alligator's Jaw during 1-week downtrend with volume spike.
Exit when price crosses the Alligator's Teeth or trend reverses.
Designed for low trade frequency by requiring strong trend alignment and volume confirmation.
Williams Alligator (SMMA-based) works well in trending markets and avoids whipsaws in ranging periods.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def smma(arr, period):
    """Smoothed Moving Average (SMMA) - same as RMA/Wilder's smoothing"""
    if len(arr) < period:
        return np.full_like(arr, np.nan, dtype=float)
    result = np.full_like(arr, np.nan, dtype=float)
    # First value is simple average
    result[period-1] = np.mean(arr[:period])
    # Subsequent values: smoothed
    for i in range(period, len(arr)):
        result[i] = (result[i-1] * (period-1) + arr[i]) / period
    return result

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Williams Alligator components (13,8,5 periods shifted by 8,5,3)
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    
    # Calculate SMMA on median price
    median_price = (high + low) / 2
    jaw = smma(median_price, jaw_period)
    teeth = smma(median_price, teeth_period)
    lips = smma(median_price, lips_period)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, jaw_period//2)  # shift by 6-7 bars
    teeth = np.roll(teeth, teeth_period//2)  # shift by 3-4 bars
    lips = np.roll(lips, lips_period//2)  # shift by 2-3 bars
    
    # Load 1-week data for trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # 20-period EMA on 1w close for trend
    close_1w = df_1w['close'].values
    ema20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Volume confirmation: current volume > 2.0x 50-period average (higher threshold for lower frequency)
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema20_1w_aligned[i]) or np.isnan(vol_ma_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_50[i]
        
        if position == 0:
            # Long: price above Jaw + 1w uptrend + volume spike
            if close[i] > jaw[i] and ema20_1w_aligned[i] > ema20_1w_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price below Jaw + 1w downtrend + volume spike
            elif close[i] < jaw[i] and ema20_1w_aligned[i] < ema20_1w_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price crosses Teeth or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price below Teeth or 1w trend turns down
                if close[i] < teeth[i] or ema20_1w_aligned[i] < ema20_1w_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price above Teeth or 1w trend turns up
                if close[i] > teeth[i] or ema20_1w_aligned[i] > ema20_1w_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Williams_Alligator_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0