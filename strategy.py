#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla pivot (R1/S1) breakout with 1-day EMA(34) trend filter and volume spike confirmation.
Trades breakouts in the direction of the daily trend only when volume exceeds 1.5x the 20-period average.
Uses fixed position sizing of 0.25 to manage risk and minimize churn.
Targets 15-35 trades/year (60-140 total over 4 years) with disciplined entry/exit to minimize fee drag.
Works in both bull and bear markets by aligning with higher timeframe trend.
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
    
    # Load 4h data for Camarilla pivot calculation - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h Camarilla pivot levels from previous day's OHLC
    # Camarilla levels based on previous day's range
    high_prev = df_4h['high'].shift(1).values  # Previous 4h bar high
    low_prev = df_4h['low'].shift(1).values    # Previous 4h bar low
    close_prev = df_4h['close'].shift(1).values # Previous 4h bar close
    
    # Calculate pivot and Camarilla levels
    pivot = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    
    # Camarilla levels
    r1 = close_prev + (range_prev * 1.1 / 12)
    s1 = close_prev - (range_prev * 1.1 / 12)
    
    # Align Camarilla levels
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Load 1d data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d EMA for trend filter (34-period)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price breaks above R1, above 1d EMA (uptrend)
            if close[i] > r1_aligned[i] and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below 1d EMA (downtrend)
            elif close[i] < s1_aligned[i] and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Camarilla level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches S1 or closes below 1d EMA
                if close[i] < s1_aligned[i] or close[i] < ema_34_1d_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches R1 or closes above 1d EMA
                if close[i] > r1_aligned[i] or close[i] > ema_34_1d_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0