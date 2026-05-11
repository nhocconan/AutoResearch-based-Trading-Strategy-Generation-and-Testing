#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Calculate daily Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Use previous day's high, low, close for Camarilla calculation
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data (avoid look-ahead)
    ph = np.roll(high_1d, 1)
    pl = np.roll(low_1d, 1)
    pc = np.roll(close_1d, 1)
    ph[0] = high_1d[0]  # first day uses same day's high
    pl[0] = low_1d[0]   # first day uses same day's low
    pc[0] = close_1d[0] # first day uses same day's close
    
    # Calculate Camarilla levels
    range_1d = ph - pl
    S1 = pc + (range_1d * 1.0 / 12)
    R1 = pc + (range_1d * 1.0 / 6)
    
    # Align to 12h timeframe
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    
    # 1-day trend using EMA34
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and volume
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(S1_aligned[i]) or np.isnan(R1_aligned[i]) or
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above R1 + 1d uptrend + volume confirmation
            if close[i] > R1_aligned[i] and close[i] > ema_34_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + 1d downtrend + volume confirmation
            elif close[i] < S1_aligned[i] and close[i] < ema_34_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below S1
            if close[i] < S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above R1
            if close[i] > R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals