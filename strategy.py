#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Pivot_Range_Reversion"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get daily data once
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    # Using previous day's OHLC (available after daily close)
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Calculate pivot and support/resistance levels
    pivot = (prev_high + prev_low + prev_close) / 3
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r1 = pivot + (range_hl * 1.1 / 12)
    r2 = pivot + (range_hl * 1.1 / 6)
    r3 = pivot + (range_hl * 1.1 / 4)
    s1 = pivot - (range_hl * 1.1 / 12)
    s2 = pivot - (range_hl * 1.1 / 6)
    s3 = pivot - (range_hl * 1.1 / 4)
    
    # Align to 12h timeframe (available after daily candle closes)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    
    # Calculate 12h RSI for overbought/oversold confirmation
    delta = np.diff(close, prepend=np.nan)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing for RSI
    def WilderSmooth(data, period):
        result = np.full_like(data, np.nan)
        if len(data) < period:
            return result
        result[period-1] = np.nanmean(data[:period])
        for i in range(period, len(data)):
            result[i] = (result[i-1] * (period-1) + data[i]) / period
        return result
    
    avg_gain = WilderSmooth(gain, 14)
    avg_loss = WilderSmooth(loss, 14)
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.where(rs == 0, 100, rsi)  # Handle division by zero
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for RSI and pivot calculation
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(pivot_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(s1_aligned[i]) or np.isnan(rsi[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Mean reversion strategy around daily pivot levels
        # Long when price touches S1 with RSI oversold (< 30)
        # Short when price touches R1 with RSI overbought (> 70)
        # Exit when price returns to pivot or opposite level
        
        if position == 0:
            # Look for mean reversion entries
            # Long: price near S1 and RSI oversold
            if low[i] <= s1_aligned[i] * 1.002 and rsi[i] < 30:  # Allow 0.2% slippage
                signals[i] = 0.25
                position = 1
            # Short: price near R1 and RSI overbought
            elif high[i] >= r1_aligned[i] * 0.998 and rsi[i] > 70:  # Allow 0.2% slippage
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long position: exit when price returns to pivot or reaches S2
            if close[i] >= pivot_aligned[i] * 0.998 or low[i] <= s2_aligned[i] * 1.002:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price returns to pivot or reaches R2
            if close[i] <= pivot_aligned[i] * 1.002 or high[i] >= r2_aligned[i] * 0.998:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Price tends to revert to daily pivot levels after touching S1/R1 with momentum exhaustion.
# Uses Camarilla pivot levels from previous day (available after daily close) for S1/R1 levels.
# Enters mean reversion trades when price touches these levels with RSI showing exhaustion (<30 for long, >70 for short).
# Exits when price returns to pivot or reaches next level (S2/R2).
# Works in both ranging and trending markets as it fades extremes.
# 12h timeframe reduces trade frequency to minimize fee drag.
# Target: 50-150 trades over 4 years (12-37/year) with discrete sizing (0.25) to control costs.