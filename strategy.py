#!/usr/bin/env python3
"""
12h_1w_Camarilla_R1S1_Breakout_Volume_Regime_Filtered_v1
Hypothesis: Use weekly timeframe for trend regime and 12h for entries.
Long when price breaks above 1w R1 with weekly uptrend (price > SMA50) and volume > 2x 20-period average.
Short when price breaks below 1w S1 with weekly downtrend (price < SMA50) and volume > 2x 20-period average.
Exit when price crosses 1w pivot point. Uses volume confirmation and regime filter to reduce false signals.
Target: 15-30 trades/year per symbol. Weekly trend filter helps in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data once for regime and Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Previous week's OHLC for Camarilla calculation
    prev_high = np.roll(high_1w, 1)
    prev_low = np.roll(low_1w, 1)
    prev_close = np.roll(close_1w, 1)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Camarilla levels: R1, S1, and pivot point (PP)
    rang = prev_high - prev_low
    r1 = prev_close + 1.1 * rang / 12
    s1 = prev_close - 1.1 * rang / 12
    pp = (prev_high + prev_low + prev_close) / 3
    
    # Weekly trend filter: price relative to SMA50
    sma50_1w = pd.Series(close_1w).rolling(window=50, min_periods=50).mean().values
    
    # Align to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    pp_aligned = align_htf_to_ltf(prices, df_1w, pp)
    sma50_1w_aligned = align_htf_to_ltf(prices, df_1w, sma50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(pp_aligned[i]) or np.isnan(sma50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = prices['close'].iloc[i]
        volume = prices['volume'].iloc[i]
        
        # Volume filter: current volume > 2.0 * 20-period average
        if i >= 20:
            vol_ma = prices['volume'].iloc[i-20:i].mean()
            volume_ok = volume > 2.0 * vol_ma
        else:
            volume_ok = False
        
        # Weekly trend filter: price > SMA50 for uptrend, price < SMA50 for downtrend
        uptrend = price > sma50_1w_aligned[i]
        downtrend = price < sma50_1w_aligned[i]
        
        if position == 0:
            # Long conditions: break above R1 + volume + weekly uptrend
            if price > r1_aligned[i] and volume_ok and uptrend:
                signals[i] = 0.25
                position = 1
            # Short conditions: break below S1 + volume + weekly downtrend
            elif price < s1_aligned[i] and volume_ok and downtrend:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses back below pivot point
            if price < pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses back above pivot point
            if price > pp_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1w_Camarilla_R1S1_Breakout_Volume_Regime_Filtered_v1"
timeframe = "12h"
leverage = 1.0