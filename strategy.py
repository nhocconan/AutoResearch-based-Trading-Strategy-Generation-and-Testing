#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla Pivot Point breakout at R1/S1 levels with 12-hour EMA(50) trend filter and volume spike confirmation.
Trades breakouts in the direction of the 12h trend only when volume exceeds 2x the 20-period average.
Uses fixed position sizing (0.25) to limit exposure and reduce trade frequency.
Designed to work in both bull and bear markets by aligning with higher timeframe trend.
Target: 15-30 trades/year (60-120 total over 4 years) to minimize fee drag.
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
    
    # Calculate 4h Camarilla Pivot Points (using previous day's OHLC)
    # For intraday, we use the previous 4h bar's data as proxy for "previous day"
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Previous bar's OHLC for pivot calculation
    prev_high = np.roll(high_4h, 1)
    prev_low = np.roll(low_4h, 1)
    prev_close = np.roll(close_4h, 1)
    prev_high[0] = high_4h[0]
    prev_low[0] = low_4h[0]
    prev_close[0] = close_4h[0]
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_ = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_ * 1.1 / 12)
    s1 = pivot - (range_ * 1.1 / 12)
    
    # Align Camarilla levels
    r1_aligned = align_htf_to_ltf(prices, df_4h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_4h, s1)
    
    # Load 12h data for trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA for trend filter (50-period)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume spike: current volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0 and vol_spike:
            # Long: price breaks above R1, above 12h EMA (uptrend)
            if close[i] > r1_aligned[i] and close[i] > ema_50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below 12h EMA (downtrend)
            elif close[i] < s1_aligned[i] and close[i] < ema_50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite level or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price touches S1 or closes below 12h EMA
                if close[i] < s1_aligned[i] or close[i] < ema_50_12h_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price touches R1 or closes above 12h EMA
                if close[i] > r1_aligned[i] or close[i] > ema_50_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0