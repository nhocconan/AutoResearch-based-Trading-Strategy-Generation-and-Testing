#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation.
Camarilla levels act as intraday support/resistance; breaks indicate momentum.
Trend filter ensures we trade with the daily trend, reducing whipsaw.
Volume confirmation avoids false breakouts.
Target: 50-150 total trades over 4 years (12-37/year).
Works in bull (breakouts with trend) and bear (breakouts against trend filtered out).
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    R3 = np.full(len(close_1d), np.nan)
    S3 = np.full(len(close_1d), np.nan)
    for i in range(len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            R3[i] = close_1d[i] + (high_1d[i] - low_1d[i]) * 1.1 / 4
            S3[i] = close_1d[i] - (high_1d[i] - low_1d[i]) * 1.1 / 4
    
    # 1d EMA34 for trend filter
    ema34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1d)):
            ema34_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema34_1d[i-1]
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    
    # Align 1d indicators to 12h timeframe
    R3_12h = align_htf_to_ltf(prices, df_1d, R3)
    S3_12h = align_htf_to_ltf(prices, df_1d, S3)
    ema34_12h = align_htf_to_ltf(prices, df_1d, ema34_1d)
    vol_sma20_12h = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # warmup for EMA34 and volume SMA20
    
    for i in range(start_idx, n):
        if np.isnan(R3_12h[i]) or np.isnan(S3_12h[i]) or np.isnan(ema34_12h[i]) or np.isnan(vol_sma20_12h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 12h volume
        # Approximate 12h volume from 1d: 1d volume / 2 (since 24h/12h = 2)
        vol_12h_approx = vol_sma20_12h[i] / 2.0
        volume_confirm = volume[i] > 1.5 * vol_12h_approx
        
        if position == 0:
            # Long: Break above R3 with uptrend and volume confirmation
            if high[i] > R3_12h[i] and close[i] > ema34_12h[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Break below S3 with downtrend and volume confirmation
            elif low[i] < S3_12h[i] and close[i] < ema34_12h[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price re-enters Camarilla body (between HLC and L3) or trend reversal
            # Calculate HLC and L3 for exit condition
            HLC = close_1d[-1] if len(close_1d) > 0 else close[i]  # placeholder, use current approximation
            L3 = close_1d[-1] - (high_1d[-1] - low_1d[-1]) * 1.1 / 6 if len(close_1d) > 0 else close[i]
            # Simplified: exit if price closes below EMA34 or re-enters below R3
            if close[i] < ema34_12h[i] or close[i] < R3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price re-enters Camarilla body or trend reversal
            if close[i] > ema34_12h[i] or close[i] > S3_12h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals