#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Elder Ray Index with weekly trend filter.
# Elder Ray measures bull/bear power using EMA13. Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Weekly EMA34 determines trend: price above EMA34 = bullish trend, below = bearish trend.
# In weekly bullish trend: long when Bull Power > 0 and rising, short when Bear Power < 0 and falling.
# In weekly bearish trend: short when Bear Power < 0, long when Bull Power > 0 (counter-trend).
# Volume confirmation: volume > 1.5x 20-period average.
# Target: 20-40 trades/year per symbol to stay within frequency limits.
name = "12h_ElderRay_WeeklyTrend_Volume"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate weekly EMA34 for trend
    def ema(arr, period):
        result = np.full_like(arr, np.nan)
        if len(arr) < period:
            return result
        multiplier = 2 / (period + 1)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (arr[i] - result[i-1]) * multiplier + result[i-1]
        return result
    
    ema34_1w = ema(close_1w, 34)
    weekly_uptrend = close_1w > ema34_1w
    weekly_downtrend = close_1w < ema34_1w
    
    # Get daily data for Elder Ray calculation (EMA13)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    ema13_1d = ema(close_1d, 13)
    bull_power = high - ema13_1d  # High - EMA13
    bear_power = low - ema13_1d   # Low - EMA13
    
    # Smooth Elder Ray with 2-period EMA to reduce noise
    bull_power_smooth = ema(bull_power, 2)
    bear_power_smooth = ema(bear_power, 2)
    
    # Get 12h average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align indicators to 12h timeframe
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend.astype(float))
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend.astype(float))
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_smooth)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_smooth)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(35, 20)  # Ensure EMA13, EMA34, and volume MA are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]) or 
            np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        bull_power_val = bull_power_aligned[i]
        bear_power_val = bear_power_aligned[i]
        weekly_up = weekly_uptrend_aligned[i] > 0.5
        weekly_down = weekly_downtrend_aligned[i] > 0.5
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Determine entry based on weekly trend
            if weekly_up and volume_confirmed:
                # Weekly uptrend: look for long signals from bull power
                if bull_power_val > 0 and bull_power_val > bull_power_aligned[i-1]:
                    signals[i] = 0.25
                    position = 1
            elif weekly_down and volume_confirmed:
                # Weekly downtrend: look for short signals from bear power
                if bear_power_val < 0 and bear_power_val < bear_power_aligned[i-1]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: bull power turns negative or weekly trend changes
            if bull_power_val <= 0 or not weekly_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: bear power turns positive or weekly trend changes
            if bear_power_val >= 0 or not weekly_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals