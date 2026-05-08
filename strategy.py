#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_BullBearPower_1dTrend"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data once for trend filter and EMA13
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    # 1d EMA13 for trend filter
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    trend_1d = (close_1d > ema13_1d).astype(float)
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # 1d EMA13 for Elder Ray calculations (EMA13 of close)
    ema13_1d_close = ema13_1d  # already computed
    
    # Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema13_1d_close
    bear_power = low - ema13_1d_close
    
    # Align Bull and Bear Power to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Smooth the power values with 6-period EMA to reduce noise
    bull_power_smooth = pd.Series(bull_power_aligned).ewm(span=6, adjust=False, min_periods=6).mean().values
    bear_power_smooth = pd.Series(bear_power_aligned).ewm(span=6, adjust=False, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 13  # warmup for EMA13 and smoothing
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(trend_1d_aligned[i]) or np.isnan(bull_power_smooth[i]) or np.isnan(bear_power_smooth[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Bull Power > 0 and Bear Power < 0 (both bullish) and daily uptrend
            long_cond = (bull_power_smooth[i] > 0 and bear_power_smooth[i] < 0 and trend_1d_aligned[i] > 0.5)
            
            # Short entry: Bull Power < 0 and Bear Power > 0 (both bearish) and daily downtrend
            short_cond = (bull_power_smooth[i] < 0 and bear_power_smooth[i] > 0 and trend_1d_aligned[i] < 0.5)
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bear Power becomes positive (bearish signal)
            if bear_power_smooth[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bull Power becomes positive (bullish signal)
            if bull_power_smooth[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Elder Ray's Bull/Bear Power combined with daily trend filter on 6h timeframe.
# Bull Power (High - EMA13) measures bullish strength, Bear Power (Low - EMA13) measures bearish strength.
# Long when both powers indicate bullish bias (BP>0, BP<0) with daily uptrend.
# Short when both indicate bearish bias (BP<0, BP>0) with daily downtrend.
# Exits when power signals diverge, capturing trend changes early.
# Works in bull markets (sustained bullish power) and bear markets (sustained bearish power).
# Smoothing reduces whipsaws while maintaining responsiveness.
# Target: 20-40 trades/year to balance signal quality with cost efficiency.