#!/usr/bin/env python3
"""
12h_Camarilla_R1S1_DailyTrend_VolumeSpike
Hypothesis: Uses Camarilla pivot levels (R1/S1) on 12h timeframe with daily trend filter and volume spike confirmation.
Designed to capture mean-reversion bounces at key pivot levels while respecting daily trend direction.
Works in both bull and bear markets by only taking reversals against the pivot but aligned with daily trend.
"""

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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 12h Camarilla pivot levels (based on previous day's OHLC)
    # For 12h chart, we use daily OHLC to calculate pivots
    prev_day_high = df_1d['high'].shift(1).values  # Previous day high
    prev_day_low = df_1d['low'].shift(1).values    # Previous day low
    prev_day_close = df_1d['close'].shift(1).values # Previous day close
    
    # Calculate pivots aligned to 12h timeframe
    high_pivot = align_htf_to_ltf(prices, df_1d, prev_day_high)
    low_pivot = align_htf_to_ltf(prices, df_1d, prev_day_low)
    close_pivot = align_htf_to_ltf(prices, df_1d, prev_day_close)
    
    # Camarilla levels: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    r1 = close_pivot + (high_pivot - low_pivot) * 1.1 / 12
    s1 = close_pivot - (high_pivot - low_pivot) * 1.1 / 12
    
    # Calculate volume spike (>1.8x 30-period MA)
    vol_ma_30 = pd.Series(volume).rolling(window=30, min_periods=30).mean().values
    vol_spike = volume > (1.8 * vol_ma_30)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicators to stabilize
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1[i]) or np.isnan(s1[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Trend direction from daily EMA50
        trend_up = close[i] > ema_50_1d_aligned[i]
        trend_down = close[i] < ema_50_1d_aligned[i]
        
        # Price proximity to Camarilla levels (within 0.1%)
        proximity_to_r1 = abs(high[i] - r1[i]) / r1[i] < 0.001
        proximity_to_s1 = abs(low[i] - s1[i]) / s1[i] < 0.001
        
        # Volume confirmation
        vol_confirm = vol_spike[i]
        
        # Entry logic: 
        # Long when price touches S1 in uptrend with volume
        # Short when price touches R1 in downtrend with volume
        long_entry = proximity_to_s1 and trend_up and vol_confirm
        short_entry = proximity_to_r1 and trend_down and vol_confirm
        
        # Exit logic: Return to midpoint or opposite touch
        midpoint = (r1[i] + s1[i]) / 2
        long_exit = low[i] > midpoint  # Price back above midpoint
        short_exit = high[i] < midpoint  # Price back below midpoint
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = 0.0
            position = 0
        elif short_exit and position == -1:
            signals[i] = 0.0
            position = 0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_Camarilla_R1S1_DailyTrend_VolumeSpike"
timeframe = "12h"
leverage = 1.0