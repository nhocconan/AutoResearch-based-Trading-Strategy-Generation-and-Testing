#!/usr/bin/env python3
# 6h_1d_adx_trend_follow_v1
# Hypothesis: 6-hour trend following using daily ADX trend strength and price position relative to daily EMA200.
# Enters long when ADX > 25 (strong trend) and price above daily EMA200, short when ADX > 25 and price below daily EMA200.
# Uses ADX to filter ranging markets and capture trending moves in both bull and bear markets.
# Exit when trend weakens (ADX < 20) or price crosses back below/above EMA200.
# Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_ata, align_htf_to_ltf

name = "6h_1d_adx_trend_follow_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Calculate ADX(14) on daily timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range
    tr = np.zeros(len(high_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(high_1d)):
        hl = high_1d[i] - low_1d[i]
        hc = abs(high_1d[i] - close_1d[i-1])
        lc = abs(low_1d[i] - close_1d[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Directional Movement
    plus_dm = np.zeros(len(high_1d))
    minus_dm = np.zeros(len(high_1d))
    for i in range(1, len(high_1d)):
        up = high_1d[i] - high_1d[i-1]
        down = low_1d[i-1] - low_1d[i]
        if up > down and up > 0:
            plus_dm[i] = up
        if down > up and down > 0:
            minus_dm[i] = down
    
    # Smoothed values
    def smooth(values, period):
        smoothed = np.full_like(values, np.nan)
        if len(values) < period:
            return smoothed
        smoothed[period-1] = np.sum(values[:period])
        for i in range(period, len(values)):
            smoothed[i] = smoothed[i-1] - (smoothed[i-1] / period) + values[i]
        return smoothed
    
    atr = smooth(tr, 14)
    plus_di = 100 * smooth(plus_dm, 14) / atr
    minus_di = 100 * smooth(minus_dm, 14) / atr
    
    # Avoid division by zero
    dx = np.full_like(plus_di, np.nan)
    denom = plus_di + minus_di
    dx[denom != 0] = 100 * np.abs(plus_di[denom != 0] - minus_di[denom != 0]) / denom[denom != 0]
    adx = smooth(dx, 14)
    
    # Calculate EMA200 on daily timeframe
    ema200 = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 200:
        ema200[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema200[i] = (close_1d[i] * 2 + ema200[i-1] * 198) / 200
    
    # Align to 6h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        if np.isnan(adx_aligned[i]) or np.isnan(ema200_aligned[i]):
            signals[i] = 0.0
            continue
        
        # Trend filter: ADX > 25 for strong trend
        strong_trend = adx_aligned[i] > 25
        weak_trend = adx_aligned[i] < 20
        
        if position == 1:  # Long position
            # Exit: trend weakens or price crosses below EMA200
            if weak_trend or close[i] <= ema200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend weakens or price crosses above EMA200
            if weak_trend or close[i] >= ema200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: strong trend and price above EMA200
            if strong_trend and close[i] > ema200_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: strong trend and price below EMA200
            elif strong_trend and close[i] < ema200_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals