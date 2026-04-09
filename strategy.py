#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1-week Camarilla pivot levels with volume confirmation
# Long when price breaks above weekly R4 with volume > 1.5x average 6h volume
# Short when price breaks below weekly S4 with volume confirmation
# Uses discrete position sizing 0.25 to target ~20-40 trades/year
# Weekly pivots provide structural support/resistance that works in both bull and bear markets
# Volume confirmation filters false breakouts
# 6h timeframe balances trade frequency and signal quality

name = "6h_1w_camarilla_breakout_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Camarilla pivot levels
    # Based on previous week's OHLC
    def calculate_camarilla(h, l, c):
        # Camarilla levels based on previous period
        pivot = (h + l + c) / 3
        range_ = h - l
        # Resistance levels
        r4 = c + (range_ * 1.1 / 2)
        r3 = c + (range_ * 1.1/4)
        r2 = c + (range_ * 1.1/6)
        r1 = c + (range_ * 1.1/12)
        # Support levels
        s1 = c - (range_ * 1.1/12)
        s2 = c - (range_ * 1.1/6)
        s3 = c - (range_ * 1.1/4)
        s4 = c - (range_ * 1.1/2)
        return r4, r3, r2, r1, pivot, s1, s2, s3, s4
    
    # Calculate for each week (using previous week's data)
    r4_1w = np.full(len(high_1w), np.nan)
    s4_1w = np.full(len(high_1w), np.nan)
    
    for i in range(1, len(high_1w)):  # Start from 1 to use previous week
        r4, _, _, _, _, _, _, _, s4 = calculate_camarilla(
            high_1w[i-1], low_1w[i-1], close_1w[i-1]
        )
        r4_1w[i] = r4
        s4_1w[i] = s4
    
    # Align weekly Camarilla levels to 6h timeframe
    r4_1w_aligned = align_htf_to_ltf(prices, df_1w, r4_1w)
    s4_1w_aligned = align_htf_to_ltf(prices, df_1w, s4_1w)
    
    # Calculate 6h average volume (20-period)
    vol_s = pd.Series(volume)
    avg_vol_6h = vol_s.rolling(window=20, min_periods=20).mean().values
    
    # Pre-compute session filter (08-20 UTC) for additional confirmation
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any required data is invalid
        if (np.isnan(r4_1w_aligned[i]) or np.isnan(s4_1w_aligned[i]) or
            np.isnan(avg_vol_6h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume (20-period)
        volume_confirmed = volume[i] > 1.5 * avg_vol_6h[i]
        
        if position == 1:  # Long position
            # Exit long if price falls below weekly S3 (strong support)
            if close[i] < s4_1w_aligned[i] * 1.02:  # Small buffer below S4
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit short if price rises above weekly R3 (strong resistance)
            if close[i] > r4_1w_aligned[i] * 0.98:  # Small buffer below R4
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Breakout strategy: enter on weekly R4/S4 breakout with volume confirmation
            if close[i] > r4_1w_aligned[i] and volume_confirmed and in_session[i]:
                position = 1
                signals[i] = 0.25
            elif close[i] < s4_1w_aligned[i] and volume_confirmed and in_session[i]:
                position = -1
                signals[i] = -0.25
    
    return signals