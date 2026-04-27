#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Index with weekly trend filter and volume confirmation.
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long when Bull Power > 0 and rising, Bear Power < 0 and falling, with volume > 1.5x average.
# Short when Bear Power < 0 and falling, Bull Power < 0 and falling, with volume > 1.5x average.
# Weekly trend filter: only take longs when weekly close > EMA26, shorts when weekly close < EMA26.
# Designed for ~15-25 trades/year with strong trend filtering to avoid whipsaws.

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
    if len(df_1w) < 1:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Get daily data for EMA13 calculation (more stable than 6h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on daily close for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Align daily EMA13 to 6h timeframe
    ema13_aligned = align_htf_to_ltf(prices, df_1d, ema13_1d)
    
    # Calculate Elder Ray components
    bull_power = high - ema13_aligned  # High - EMA13
    bear_power = low - ema13_aligned   # Low - EMA13
    
    # Calculate weekly EMA26 for trend filter
    ema26_1w = pd.Series(close_1w).ewm(span=26, adjust=False, min_periods=26).mean().values
    
    # Align weekly EMA26 to 6h timeframe
    ema26_aligned = align_htf_to_ltf(prices, df_1w, ema26_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need 20-period volume MA and 13-period EMA
    start_idx = max(20, 13)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema13_aligned[i]) or np.isnan(ema26_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema26_aligned[i]
        weekly_downtrend = close[i] < ema26_aligned[i]
        
        # Elder Ray conditions
        bull_power_rising = bull_power[i] > bull_power[i-1] if i > 0 else False
        bull_power_positive = bull_power[i] > 0
        bear_power_falling = bear_power[i] < bear_power[i-1] if i > 0 else False
        bear_power_negative = bear_power[i] < 0
        
        if position == 0:
            # Long: Bull Power positive AND rising, with volume and weekly uptrend
            if bull_power_positive and bull_power_rising and vol_filter and weekly_uptrend:
                signals[i] = size
                position = 1
            # Short: Bear Power negative AND falling, with volume and weekly downtrend
            elif bear_power_negative and bear_power_falling and vol_filter and weekly_downtrend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Bull Power turns negative OR weekly trend turns down
            if bull_power[i] <= 0 or not weekly_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Bear Power turns positive OR weekly trend turns up
            if bear_power[i] >= 0 or not weekly_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_ElderRay_WeeklyTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0