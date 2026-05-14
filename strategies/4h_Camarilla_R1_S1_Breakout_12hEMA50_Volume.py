#!/usr/bin/env python3

"""
Hypothesis: 4-hour Camarilla R1/S1 breakout with 12-hour trend filter and volume confirmation.
Camarilla pivot levels provide precise intraday support/resistance derived from prior period.
The 12-hour trend filter ensures trades align with intermediate-term direction, reducing counter-trend trades.
Volume spikes confirm institutional participation. This combination targets high-probability breakouts
in both bull and bear markets by focusing on volatility expansion at key levels.
Target: 20-40 trades/year per symbol.
"""

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
    
    # Load 12h data - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # Calculate 12h EMA for trend filter (50-period)
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate daily Camarilla pivot levels (R1, S1)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    prev_open = df_1d['open'].shift(1).values
    
    # True range for Camarilla (using close-close as base)
    cam_base = np.abs(prev_close - prev_open)
    # Prevent division by zero
    cam_base = np.where(cam_base == 0, (prev_high - prev_low) * 0.1, cam_base)
    
    # Camarilla R1 and S1 levels
    r1 = prev_close + cam_base * 1.1 / 12
    s1 = prev_close - cam_base * 1.1 / 12
    
    # Align daily Camarilla levels to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 4h volume average (20-period)
    vol_avg_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Pre-calculate session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(vol_avg_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above R1, above 12h EMA, volume spike
            if (close[i] > r1_aligned[i] and    # Break above Camarilla R1
                close[i] > ema_50_aligned[i] and # Above 12h EMA (bullish trend)
                volume[i] > 2.0 * vol_avg_20[i]): # Volume spike
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1, below 12h EMA, volume spike
            elif (close[i] < s1_aligned[i] and   # Break below Camarilla S1
                  close[i] < ema_50_aligned[i] and # Below 12h EMA (bearish trend)
                  volume[i] > 2.0 * vol_avg_20[i]): # Volume spike
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price returns to opposite Camarilla level or crosses 12h EMA
            exit_signal = False
            
            if position == 1:
                # Exit long: price crosses below S1 or below 12h EMA
                if close[i] < s1_aligned[i] or close[i] < ema_50_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price crosses above R1 or above 12h EMA
                if close[i] > r1_aligned[i] or close[i] > ema_50_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R1_S1_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0