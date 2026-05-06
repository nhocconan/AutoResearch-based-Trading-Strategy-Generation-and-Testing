#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-day Camarilla pivot levels with volume confirmation
# Long when price breaks above R4 level with volume > 1.5x average (bullish breakout)
# Short when price breaks below S4 level with volume > 1.5x average (bearish breakout)
# Camarilla pivot levels provide clear support/resistance zones from prior day,
# Volume confirms breakout strength. Works in both bull/bear markets by capturing
# genuine breakouts at extreme levels. Target: 12-37 trades/year (50-150 over 4 years).

name = "6h_1dCamarilla_R4S4_Volume_Breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day Camarilla pivot levels ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Calculate pivot point
    pivot = (prev_high + prev_low + prev_close) / 3.0
    
    # Calculate Camarilla levels
    R4 = pivot + (prev_high - prev_low) * 1.1 / 2.0
    R3 = pivot + (prev_high - prev_low) * 1.1 / 4.0
    S3 = pivot - (prev_high - prev_low) * 1.1 / 4.0
    S4 = pivot - (prev_high - prev_low) * 1.1 / 2.0
    
    # Align Camarilla levels to 6h timeframe
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    
    # Volume confirmation: >1.5x 50-period average
    vol_ma_50 = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    volume_filter = volume > (1.5 * vol_ma_50)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any critical value is NaN or outside session
        if (np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(volume_filter[i]) or not session_filter[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long breakout: price breaks above R4 with volume confirmation
            if close[i] > R4_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short breakout: price breaks below S4 with volume confirmation
            elif close[i] < S4_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 (mean reversion)
            if close[i] < S3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 (mean reversion)
            if close[i] > R3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals