#!/usr/bin/env python3
# 6h_1d_camarilla_pivot_v1
# Hypothesis: Camarilla pivot levels from 1-day timeframe to identify mean reversion and breakout opportunities.
# Fade at R3/S3 levels (strong mean reversion zones) and continue trend at R4/S4 breakouts.
# Uses 6-hour timeframe for execution with 1-day Camarilla pivots as support/resistance.
# Works in both bull and bear markets by adapting to volatility and price extremes.
# Target: 20-40 trades/year to minimize fee drag while capturing high-probability setups.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1d_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First day has no previous data
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Calculate pivot and ranges
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    r4 = pivot + range_hl * 1.1 / 2
    r3 = pivot + range_hl * 1.1 / 4
    s3 = pivot - range_hl * 1.1 / 4
    s4 = pivot - range_hl * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    
    # Calculate 6h RSI for overbought/oversold confirmation
    delta = np.diff(close)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros(n)
    avg_loss = np.zeros(n)
    avg_gain[14] = np.mean(gain[:14]) if n > 14 else 0
    avg_loss[14] = np.mean(loss[:14]) if n > 14 else 0
    
    for i in range(15, n):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        price = close[i]
        r3 = r3_aligned[i]
        s3 = s3_aligned[i]
        r4 = r4_aligned[i]
        s4 = s4_aligned[i]
        rsi_val = rsi[i]
        
        if np.isnan(r3) or np.isnan(s3) or np.isnan(r4) or np.isnan(s4) or np.isnan(rsi_val):
            if position != 0:
                pass  # Hold
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long
            # Exit: price breaks above R4 (take profit) or falls below S3 (stop)
            if price >= r4 or price <= s3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short
            # Exit: price breaks below S4 (take profit) or rises above R3 (stop)
            if price <= s4 or price >= r3:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Fade at S3 (long) when oversold
            if price <= s3 and rsi_val < 30:
                position = 1
                signals[i] = 0.25
            # Fade at R3 (short) when overbought
            elif price >= r3 and rsi_val > 70:
                position = -1
                signals[i] = -0.25
            # Breakout continuation: long above R4
            elif price > r4 and rsi_val > 50:
                position = 1
                signals[i] = 0.25
            # Breakout continuation: short below S4
            elif price < s4 and rsi_val < 50:
                position = -1
                signals[i] = -0.25
    
    return signals