#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Weekly Momentum Filter
# Uses Elder Ray (Bull/Bear Power) from daily data to identify institutional buying/selling pressure.
# Filters trades with weekly RSI momentum: only take long signals when weekly RSI > 50 (bullish bias),
# and short signals when weekly RSI < 50 (bearish bias). Works in bull markets (buy strength) and
# bear markets (sell weakness). Target: 50-150 total trades over 4 years.

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
    if len(df_1d) < 22:  # Need at least ~1 month for EMA
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Load weekly data for momentum filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Calculate Elder Ray components (13-period EMA as in original)
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_1d - ema13_1d  # Bull Power: High - EMA13
    bear_power = low_1d - ema13_1d   # Bear Power: Low - EMA13
    
    # Calculate weekly RSI (14-period)
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi_1w = 100 - (100 / (1 + rs))
    
    # Align Elder Bull/Bear Power to 6h timeframe
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    # Align weekly RSI to 6f timeframe (no extra delay needed for RSI)
    rsi_1w_aligned = align_htf_to_ltf(prices, df_1w, rsi_1w)
    
    signals = np.zeros(n)
    position = 0
    base_size = 0.25  # Position size (25% of capital)
    
    for i in range(100, n):
        # Skip if any required data is NaN
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or
            np.isnan(rsi_1w_aligned[i])):
            continue
        
        # Long entry: Bull Power > 0 (buying pressure) + weekly RSI > 50 (bullish momentum)
        if (bull_power_aligned[i] > 0 and
            rsi_1w_aligned[i] > 50 and
            position <= 0):
            position = 1
            signals[i] = base_size
        
        # Short entry: Bear Power < 0 (selling pressure) + weekly RSI < 50 (bearish momentum)
        elif (bear_power_aligned[i] < 0 and
              rsi_1w_aligned[i] < 50 and
              position >= 0):
            position = -1
            signals[i] = -base_size
        
        # Exit: Opposite signal appears
        elif position == 1 and bear_power_aligned[i] < 0:
            position = 0
            signals[i] = 0.0
        elif position == -1 and bull_power_aligned[i] > 0:
            position = 0
            signals[i] = 0.0
    
    return signals

name = "6h_ElderRay_WeeklyRSI_Filter"
timeframe = "6h"
leverage = 1.0