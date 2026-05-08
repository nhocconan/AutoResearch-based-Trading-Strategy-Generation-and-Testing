#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot point reversal with 1d EMA34 trend filter and volume spike confirmation.
# Long when price touches or crosses below Camarilla S1 level AND 1d EMA34 rising AND volume > 1.5x 20-period average.
# Short when price touches or crosses above Camarilla R1 level AND 1d EMA34 falling AND volume > 1.5x 20-period average.
# Exit when price crosses back to the pivot point (mean reversion to daily equilibrium).
# This strategy targets mean reversion in established trends with volume confirmation to avoid false reversals.
# Camarilla levels provide precise support/resistance levels derived from daily range.
# The 1d EMA34 filter ensures we trade reversals only in the direction of the higher timeframe trend.
# Volume spike confirms institutional participation in the reversal.
# Target: 100-200 total trades over 4 years (25-50/year).

name = "4h_Camarilla_S1R1_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day's range
    high_prev = df_1d['high'].shift(1).values
    low_prev = df_1d['low'].shift(1).values
    close_prev = df_1d['close'].shift(1).values
    
    # Pivot point calculation
    pivot = (high_prev + low_prev + close_prev) / 3
    range_prev = high_prev - low_prev
    
    # Camarilla levels: S1, R1 (most important for intraday reversals)
    camarilla_s1 = close_prev - (range_prev * 1.1 / 12)
    camarilla_r1 = close_prev + (range_prev * 1.1 / 12)
    
    # Align Camarilla levels to 4h timeframe
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34)  # Sufficient warmup for EMA34
    
    for i in range(start_idx, n):
        if (np.isnan(camarilla_s1_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(pivot_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(ema34_rising[i]) or np.isnan(ema34_falling[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price touches or crosses below S1, EMA34 rising, volume filter
            long_cond = (low[i] <= camarilla_s1_aligned[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price touches or crosses above R1, EMA34 falling, volume filter
            short_cond = (high[i] >= camarilla_r1_aligned[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses back to pivot point (mean reversion)
            if close[i] >= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses back to pivot point (mean reversion)
            if close[i] <= pivot_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals