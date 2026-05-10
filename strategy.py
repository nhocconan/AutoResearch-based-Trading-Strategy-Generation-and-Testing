#!/usr/bin/env python3
# 1D_Weekly_Camarilla_R1_S1_Breakout_With_Volume
# Hypothesis: Weekly Camarilla pivot levels act as strong support/resistance zones.
# Breakout above R1 or below S1 with volume confirmation (>1.5x 20-day avg) captures institutional moves.
# Weekly trend filter (price > weekly EMA20) ensures alignment with higher timeframe momentum.
# Designed for low trade frequency (~10-20/year) with discrete sizing (0.25) to minimize fee drag.
# Works in bull markets (breakouts with trend) and bear markets (breakdowns against trend).

name = "1D_Weekly_Camarilla_R1_S1_Breakout_With_Volume"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Weekly Camarilla pivot levels (using prior week's OHLC)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Camarilla levels: R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    camarilla_width = 1.1 * (high_1w - low_1w) / 12
    r1_1w = close_1w + camarilla_width
    s1_1w = close_1w - camarilla_width
    
    # Align weekly levels to daily timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # Weekly trend filter: EMA 20
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough history for indicators
    
    for i in range(start_idx, n):
        if np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) or np.isnan(ema_20_1w_aligned[i]) or np.isnan(vol_threshold[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        is_uptrend = close[i] > ema_20_1w_aligned[i]
        is_downtrend = close[i] < ema_20_1w_aligned[i]
        
        if position == 0:
            # Long entry: Price breaks above weekly R1 + volume confirmation + weekly uptrend
            if close[i] > r1_1w_aligned[i] and volume[i] > vol_threshold[i] and is_uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below weekly S1 + volume confirmation + weekly downtrend
            elif close[i] < s1_1w_aligned[i] and volume[i] > vol_threshold[i] and is_downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below weekly S1 or weekly trend turns down
            if close[i] < s1_1w_aligned[i] or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above weekly R1 or weekly trend turns up
            if close[i] > r1_1w_aligned[i] or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals