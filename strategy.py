# 6h_WeeklyPivot_TrendFilter_WithVolume
# Hypothesis: Weekly pivot levels provide strong support/resistance in both bull and bear markets.
# Price breaking above weekly R1 with volume indicates bullish momentum; breaking below S1 indicates bearish momentum.
# Weekly timeframe filters noise, 6h provides timely entries, volume confirms institutional participation.
# Target: 50-150 trades over 4 years (12-37/year) with size 0.25.

#!/usr/bin/env python3
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
    
    # Get weekly data for pivot levels (primary HTF)
    weekly = get_htf_data(prices, '1w')
    weekly_high = weekly['high'].values
    weekly_low = weekly['low'].values
    weekly_close = weekly['close'].values
    
    # Calculate weekly pivot levels (standard floor trader formula)
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - weekly_low
    weekly_s1 = 2 * weekly_pivot - weekly_high
    weekly_r2 = weekly_pivot + (weekly_high - weekly_low)
    weekly_s2 = weekly_pivot - (weekly_high - weekly_low)
    
    # Align weekly pivot levels to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, weekly, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, weekly, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, weekly, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, weekly, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, weekly, weekly_s2)
    
    # Get daily data for trend filter (secondary HTF)
    daily = get_htf_data(prices, '1d')
    daily_close = daily['close'].values
    # Calculate 50-period EMA on daily close
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema50_aligned = align_htf_to_ltf(prices, daily, daily_ema50)
    
    # Volume filter: current 6h volume > 2.0x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(weekly_r1_aligned[i]) or 
            np.isnan(weekly_s1_aligned[i]) or np.isnan(daily_ema50_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Only trade when volume filter passes
        if volume_filter[i]:
            # Long conditions: price breaks above weekly R1 AND above daily EMA50 (uptrend)
            if close[i] > weekly_r1_aligned[i] and close[i] > daily_ema50_aligned[i]:
                signals[i] = 0.25
            # Long conditions: price bounces from weekly S1 with volume (above S1, below S2) AND above daily EMA50
            elif close[i] > weekly_s1_aligned[i] and close[i] < weekly_s2_aligned[i] and close[i] > daily_ema50_aligned[i]:
                signals[i] = 0.25
            # Short conditions: price breaks below weekly S1 AND below daily EMA50 (downtrend)
            elif close[i] < weekly_s1_aligned[i] and close[i] < daily_ema50_aligned[i]:
                signals[i] = -0.25
            # Short conditions: price rejected at weekly R1 with volume (below R1, above R2) AND below daily EMA50
            elif close[i] < weekly_r1_aligned[i] and close[i] > weekly_r2_aligned[i] and close[i] < daily_ema50_aligned[i]:
                signals[i] = -0.25
            else:
                signals[i] = signals[i-1]
        else:
            signals[i] = signals[i-1]
    
    return signals

name = "6h_WeeklyPivot_TrendFilter_WithVolume"
timeframe = "6h"
leverage = 1.0