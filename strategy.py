#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot reversal with 1d trend filter and volume confirmation.
# Long when price touches S1 and rebounds, 1d EMA34 rising, volume > 2x 24-period average.
# Short when price touches R1 and reverses, 1d EMA34 falling, volume > 2x 24-period average.
# Exit when price crosses the 12-period EMA on 12h chart.
# Uses Camarilla pivot levels from daily data for high-probability reversal zones.
# EMA34 filter ensures trading with the higher timeframe trend.
# Volume spike confirms institutional interest at pivot levels.
# Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Camarilla_S1R1_1dEMA34_Volume"
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
    
    # 1d data for Camarilla pivot and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Using previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    # Camarilla levels
    s1 = pivot - 1.1 * (prev_high - prev_low) / 12.0
    r1 = pivot + 1.1 * (prev_high - prev_low) / 12.0
    
    # Align Camarilla levels to 12h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    
    # 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 1d EMA34 direction
    ema34_rising = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_falling = np.zeros_like(ema34_1d_aligned, dtype=bool)
    ema34_rising[1:] = ema34_1d_aligned[1:] > ema34_1d_aligned[:-1]
    ema34_falling[1:] = ema34_1d_aligned[1:] < ema34_1d_aligned[:-1]
    
    # Volume filter: current volume > 2x 24-period average (adjust for 12h)
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > (2.0 * vol_ma24)
    
    # 12h EMA12 for exit
    ema12 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 34, 24)  # Sufficient warmup
    
    for i in range(start_idx, n):
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(ema34_1d_aligned[i]) or np.isnan(ema34_rising[i]) or 
            np.isnan(ema34_falling[i]) or np.isnan(volume_filter[i]) or
            np.isnan(ema12[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price touches S1 and rebounds (close > S1), EMA34 rising, volume filter
            long_cond = (close[i] > s1_aligned[i]) and (low[i] <= s1_aligned[i] * 1.001) and ema34_rising[i] and volume_filter[i]
            # Short conditions: price touches R1 and reverses (close < R1), EMA34 falling, volume filter
            short_cond = (close[i] < r1_aligned[i]) and (high[i] >= r1_aligned[i] * 0.999) and ema34_falling[i] and volume_filter[i]
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below 12h EMA12
            if close[i] < ema12[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above 12h EMA12
            if close[i] > ema12[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals