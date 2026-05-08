#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot reversal with 1d trend filter and volume spike confirmation.
# Long when price touches Camarilla S1 level AND 1d EMA34 rising AND volume > 2x 20-period average.
# Short when price touches Camarilla R1 level AND 1d EMA34 falling AND volume > 2x 20-period average.
# Exit when price reverses to Camarilla Pivot level or opposite S/R level.
# This strategy captures mean reversals at key pivot levels with trend alignment and volume confirmation.
# Camarilla levels provide precise support/resistance. The 1d EMA34 filter ensures we trade with higher timeframe trend.
# Volume spike confirms institutional participation at reversal points. Target: 75-200 total trades over 4 years (19-50/year).

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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # Using typical Camarilla formula based on previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Camarilla levels
    S1 = close - (range_val * 1.1 / 12)  # Support level 1
    R1 = close + (range_val * 1.1 / 12)  # Resistance level 1
    
    # Align Camarilla levels to 4h timeframe
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 2x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (2.0 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # Sufficient warmup for EMA34 and Camarilla
    
    for i in range(start_idx, n):
        if (np.isnan(S1_aligned[i]) or np.isnan(R1_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_rising[i]) or 
            np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price touches S1 level, 1d EMA34 rising, volume filter
            long_cond = (low[i] <= S1_aligned[i]) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price touches R1 level, 1d EMA34 falling, volume filter
            short_cond = (high[i] >= R1_aligned[i]) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price reverses to pivot or above R1
            if close[i] >= pivot_aligned[i] or high[i] >= R1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price reverses to pivot or below S1
            if close[i] <= pivot_aligned[i] or low[i] <= S1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals