#!/usr/bin/env python3
# 12H_Camarilla_R1_S1_1DTrend_VolumeBreakout
# Hypothesis: 12-hour strategy using daily Camarilla R1/S1 breakouts with 1-day EMA34 trend filter and volume spike confirmation.
# Targets only the strongest breakouts in the direction of the daily trend to reduce false signals.
# Works in both bull and bear markets by using trend filter to avoid counter-trend trades.
# Targets 15-35 trades/year to minimize fee drag. Uses discrete position sizing (0.25).

name = "12H_Camarilla_R1_S1_1DTrend_VolumeBreakout"
timeframe = "12h"
leverage = 1.0

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
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:  # Need enough data for EMA34
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day's OHLC)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate Camarilla levels for each day
    range_1d = prev_high - prev_low
    r1 = prev_close + range_1d * 1.1 / 12
    s1 = prev_close - range_1d * 1.1 / 12
    
    # Calculate 1-day EMA34 for trend filter
    ema_34 = pd.Series(prev_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align Camarilla levels and EMA to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Volume filter: current volume > 2.0x average volume (30-period)
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 30)  # Ensure we have EMA34 and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (2.0x average volume)
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R1 + uptrend (price > EMA34) + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_34_aligned[i] and   # Uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + downtrend (price < EMA34) + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_34_aligned[i] and   # Downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price returns to the opposite S1/R1 level (mean reversion)
            if (position == 1 and close[i] < s1_aligned[i]) or \
               (position == -1 and close[i] > r1_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals