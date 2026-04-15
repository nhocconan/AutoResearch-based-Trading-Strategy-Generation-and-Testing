#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1w Trend Filter
# Uses daily Bull/Bear Power (EMA13) to measure buying/selling pressure.
# Weekly trend filter (EMA50) ensures we trade with the higher timeframe trend.
# Works in bull markets (buy when Bull Power > 0 and price above weekly EMA50)
# and bear markets (sell when Bear Power < 0 and price below weekly EMA50).
# Target: 50-150 total trades over 4 years.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate EMA13 on daily close for Elder Ray
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Bull Power = Daily High - EMA13
    bull_power_1d = high_1d - ema13_1d
    # Bear Power = Daily Low - EMA13
    bear_power_1d = low_1d - ema13_1d
    
    # Calculate EMA50 on weekly close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align indicators to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(ema50_1w_aligned[i])):
            continue
        
        # Long entry: Bull Power > 0 (buying pressure) + price above weekly EMA50 (uptrend)
        if (bull_power_aligned[i] > 0 and
            close[i] > ema50_1w_aligned[i] and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bear Power < 0 (selling pressure) + price below weekly EMA50 (downtrend)
        elif (bear_power_aligned[i] < 0 and
              close[i] < ema50_1w_aligned[i] and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: opposite signal or loss of trend
        elif position == 1 and (bear_power_aligned[i] < 0 or close[i] < ema50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (bull_power_aligned[i] > 0 or close[i] > ema50_1w_aligned[i]):
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_1wTrend"
timeframe = "6h"
leverage = 1.0