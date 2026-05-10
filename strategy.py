#!/usr/bin/env python3
# 1d_Weekly_Camarilla_R1_S1_Breakout_With_Volume
# Hypothesis: Camarilla pivot levels (R1/S1) on weekly timeframe act as strong support/resistance.
# Breaks above R1 with volume confirmation indicate bullish momentum; breaks below S1 with volume indicate bearish momentum.
# Weekly trend filter (price above/below weekly EMA20) ensures alignment with higher timeframe trend.
# Designed for low trade frequency (~10-20/year) with discrete sizing (0.25) to minimize fee drift.
# Works in both bull and bear markets by following the weekly trend.

name = "1d_Weekly_Camarilla_R1_S1_Breakout_With_Volume"
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
    
    # Weekly data for Camarilla pivot calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from weekly OHLC
    # R1 = close + (high - low) * 1.1 / 12
    # S1 = close - (high - low) * 1.1 / 12
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    camarilla_width = (weekly_high - weekly_low) * 1.1 / 12.0
    r1 = weekly_close + camarilla_width
    s1 = weekly_close - camarilla_width
    
    # Align weekly R1/S1 to daily timeframe (wait for weekly bar to close)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Weekly trend filter: EMA 20 on weekly close
    weekly_ema20 = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    weekly_ema20_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema20)
    
    # Volume confirmation: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_threshold = vol_ma * 1.5
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough history for volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(weekly_ema20_aligned[i]) or np.isnan(vol_threshold[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long entry: Price breaks above weekly R1 + volume confirmation + weekly uptrend
            if close[i] > r1_aligned[i] and volume[i] > vol_threshold[i] and close[i] > weekly_ema20_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short entry: Price breaks below weekly S1 + volume confirmation + weekly downtrend
            elif close[i] < s1_aligned[i] and volume[i] > vol_threshold[i] and close[i] < weekly_ema20_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price crosses below weekly S1 (mean reversion to pivot)
            if close[i] < s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above weekly R1 (mean reversion to pivot)
            if close[i] > r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals