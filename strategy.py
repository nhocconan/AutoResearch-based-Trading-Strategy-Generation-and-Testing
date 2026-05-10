#!/usr/bin/env python3
# 1D_Weekly_Camarilla_R1_S1_Breakout_With_Volume
# Hypothesis: Daily Camarilla pivot levels act as key support/resistance in both bull and bear markets.
# Breakouts above R1 or below S1 with volume confirmation indicate strong momentum.
# Weekly trend filter (price above/below weekly EMA50) ensures alignment with higher timeframe trend.
# Designed for low trade frequency (~10-25/year) with discrete sizing (0.25) to minimize fee drag.
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend).

name = "1D_Weekly_Camarilla_R1_S1_Breakout_With_Volume"
timeframe = "1d"
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
    
    # Calculate Camarilla pivot levels for each day
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    camarilla_range = high - low
    r1 = close + camarilla_range * 1.1 / 12
    s1 = close - camarilla_range * 1.1 / 12
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    # Weekly trend filter: EMA 50 on weekly data
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(r1[i]) or np.isnan(s1[i]) or np.isnan(vol_threshold[i]) or np.isnan(ema_50_1w_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Price breaks above R1 + volume confirmation + weekly uptrend
            if close[i] > r1[i] and volume[i] > vol_threshold[i] and close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below S1 + volume confirmation + weekly downtrend
            elif close[i] < s1[i] and volume[i] > vol_threshold[i] and close[i] < ema_50_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below S1 or weekly trend turns down
            if close[i] < s1[i] or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above R1 or weekly trend turns up
            if close[i] > r1[i] or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals