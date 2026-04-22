#!/usr/bin/env python3
"""
Hypothesis: 4-hour Camarilla Pivot Reversal with 1-day Trend Filter and Volume Spike.
Long when price touches S1 during 1-day uptrend with volume spike; short when price touches R1 during 1-day downtrend with volume spike.
Exit when price reaches midpoint or trend reverses.
Designed for low trade frequency (target: 20-50/year) by requiring confluence of pivot level, trend, and volume.
Works in both bull and bear markets by following 1-day trend and using mean-reversion at key levels.
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
    
    # Calculate Camarilla pivot levels from previous day
    # Typical price = (H + L + C) / 3
    typical_price = (high + low + close) / 3
    # Range = H - L
    daily_range = high - low
    # Camarilla levels
    R4 = close + (1.1/1.2) * daily_range
    R3 = close + (1.1/6) * daily_range
    R2 = close + (1.1/4) * daily_range
    R1 = close + (1.1/12) * daily_range
    S1 = close - (1.1/12) * daily_range
    S2 = close - (1.1/4) * daily_range
    S3 = close - (1.1/6) * daily_range
    S4 = close - (1.1/1.2) * daily_range
    Pivot = (high + low + close) / 3
    
    # Shift levels to use previous day's values
    R1 = np.roll(R1, 1)
    S1 = np.roll(S1, 1)
    Pivot = np.roll(Pivot, 1)
    R1[0] = np.nan
    S1[0] = np.nan
    Pivot[0] = np.nan
    
    # Load 1-day data for trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # 34-period EMA on 1d close for trend
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(40, n):
        # Skip if data not ready
        if (np.isnan(R1[i]) or np.isnan(S1[i]) or np.isnan(Pivot[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: price touches S1 + 1d uptrend + volume spike
            if low[i] <= S1[i] and ema34_1d_aligned[i] > ema34_1d_aligned[i-1] and vol_spike:
                signals[i] = 0.25
                position = 1
            # Short: price touches R1 + 1d downtrend + volume spike
            elif high[i] >= R1[i] and ema34_1d_aligned[i] < ema34_1d_aligned[i-1] and vol_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: price reaches midpoint or trend reverses
            exit_signal = False
            
            if position == 1:
                # Exit long: price >= pivot or 1d trend turns down
                if high[i] >= Pivot[i] or ema34_1d_aligned[i] < ema34_1d_aligned[i-1]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: price <= pivot or 1d trend turns up
                if low[i] <= Pivot[i] or ema34_1d_aligned[i] > ema34_1d_aligned[i-1]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Camarilla_Pivot_Reversal_1dTrend_VolumeSpike"
timeframe = "4h"
leverage = 1.0