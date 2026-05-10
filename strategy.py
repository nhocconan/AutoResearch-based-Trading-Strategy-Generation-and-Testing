#!/usr/bin/env python3
# 1D_1W_KAMA_Trend_Filter_With_Volume_Spike
# Hypothesis: On 1d timeframe, enter long when KAMA turns bullish (price > KAMA) with weekly uptrend and volume spike (>2x 20-day average).
# Enter short when KAMA turns bearish (price < KAMA) with weekly downtrend and volume spike.
# Uses weekly trend filter to avoid counter-trend trades and volume spike for confirmation.
# Target: 15-25 trades/year per symbol (60-100 total over 4 years).

name = "1D_1W_KAMA_Trend_Filter_With_Volume_Spike"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate KAMA (Kaufman Adaptive Moving Average) on daily close
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close, n=10))
    volatility = np.sum(np.abs(np.diff(close, n=1)), axis=0)
    # Handle first 10 values
    er = np.full_like(change, np.nan, dtype=float)
    er[10:] = change[10:] / np.maximum(volatility[10:], 1e-10)
    # Smoothing constants
    sc = (er * (2/2 - 2/30) + 2/30) ** 2  # fast=2, slow=30
    # Initialize KAMA
    kama = np.full_like(close, np.nan, dtype=float)
    kama[9] = close[9]  # start at index 9
    for i in range(10, len(close)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    # KAMA trend: price above/below KAMA
    kama_bullish = close > kama
    kama_bearish = close < kama
    
    # Weekly trend: EMA(21) on weekly close
    ema_21_1w = pd.Series(close_1w).ewm(span=21, adjust=False, min_periods=21).mean().values
    weekly_uptrend = close_1w > ema_21_1w
    weekly_downtrend = close_1w < ema_21_1w
    
    # Volume spike: current volume > 2x 20-day average
    volume_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_avg * 2.0)
    
    # Align weekly indicators to daily
    weekly_uptrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_uptrend)
    weekly_downtrend_aligned = align_htf_to_ltf(prices, df_1w, weekly_downtrend)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if np.isnan(kama[i]) or np.isnan(weekly_uptrend_aligned[i]) or np.isnan(weekly_downtrend_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price > KAMA (bullish) + weekly uptrend + volume spike
            if kama_bullish[i] and weekly_uptrend_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price < KAMA (bearish) + weekly downtrend + volume spike
            elif kama_bearish[i] and weekly_downtrend_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price < KAMA (turn bearish) or weekly trend changes
            if kama_bearish[i] or not weekly_uptrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price > KAMA (turn bullish) or weekly trend changes
            if kama_bullish[i] or not weekly_downtrend_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals