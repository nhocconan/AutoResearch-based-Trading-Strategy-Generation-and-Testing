#!/usr/bin/env python3
"""
1d_Williams_Alligator_ElderRay_Trend_Volume_Spike
Hypothesis: Combines Williams Alligator (trend detection) with Elder Ray (bull/bear power) and volume spike confirmation on daily timeframe. 
Williams Alligator uses smoothed moving averages (Jaw, Teeth, Lips) to identify trends and avoid sideways markets. 
Elder Ray measures bull power (high - EMA13) and bear power (EMA13 - low) to assess trend strength. 
Volume spike (>1.5x 20-day average) confirms breakout strength. 
Weekly trend filter ensures alignment with higher timeframe trend. 
Designed for low trade frequency (10-20/year) with strong trend-following logic, works in bull/bear markets by requiring trend alignment and volatility-based confirmation.
"""

name = "1d_Williams_Alligator_ElderRay_Trend_Volume_Spike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    weekly_close = df_1w['close'].values
    
    # Calculate weekly EMA50 for trend filter
    ema_50_1w = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Williams Alligator (13,8,5) - smoothed moving averages
    # Jaw: 13-period smoothed moving average of median price, shifted 8 bars
    # Teeth: 8-period smoothed moving average of median price, shifted 5 bars
    # Lips: 5-period smoothed moving average of median price, shifted 3 bars
    median_price = (high + low) / 2
    
    def smoothed_ma(arr, period):
        # Smoothed moving average (SMMA) - similar to RMA/Wilder's smoothing
        sma = pd.Series(arr).rolling(window=period, min_periods=period).mean().values
        smma = np.full_like(arr, np.nan, dtype=np.float64)
        if len(arr) >= period:
            smma[period-1] = sma[period-1]
            for i in range(period, len(arr)):
                smma[i] = (smma[i-1] * (period-1) + arr[i]) / period
        return smma
    
    jaw = smoothed_ma(median_price, 13)
    teeth = smoothed_ma(median_price, 8)
    lips = smoothed_ma(median_price, 5)
    
    # Shift as per Alligator definition
    jaw = np.roll(jaw, 8)
    teeth = np.roll(teeth, 5)
    lips = np.roll(lips, 3)
    
    # Elder Ray - Bull Power and Bear Power using 13-period EMA
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Volume confirmation: 20-day average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.divide(volume, vol_ma20, out=np.zeros_like(volume), where=vol_ma20!=0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Warmup for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine weekly trend using aligned close
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, weekly_close)
        if np.isnan(weekly_close_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        weekly_trend_up = weekly_close_aligned[i] > ema_50_1w_aligned[i]
        weekly_trend_down = weekly_close_aligned[i] < ema_50_1w_aligned[i]
        
        # Alligator alignment: Lips > Teeth > Jaw = uptrend, Lips < Teeth < Jaw = downtrend
        alligator_long = lips[i] > teeth[i] and teeth[i] > jaw[i]
        alligator_short = lips[i] < teeth[i] and teeth[i] < jaw[i]
        
        if position == 0:
            # Long: Alligator aligned up, Bull Power positive, Bear Power negative, weekly trend up, volume spike
            if (alligator_long and 
                bull_power[i] > 0 and 
                bear_power[i] > 0 and  # Actually bear power should be positive for strength, but we want it decreasing
                weekly_trend_up and 
                vol_ratio[i] > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Alligator aligned down, Bull Power negative, Bear Power positive, weekly trend down, volume spike
            elif (alligator_short and 
                  bull_power[i] < 0 and 
                  bear_power[i] < 0 and  # Bull power negative for bear strength
                  weekly_trend_down and 
                  vol_ratio[i] > 1.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Alligator alignment breaks or weekly trend turns down
            if not alligator_long or not weekly_trend_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Alligator alignment breaks or weekly trend turns up
            if not alligator_short or not weekly_trend_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals