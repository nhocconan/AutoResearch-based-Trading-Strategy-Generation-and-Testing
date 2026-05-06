#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d KAMA for trend direction and 1w Williams %R for mean reversion in ranging markets
# - Uses 1d Kaufman Adaptive Moving Average (KAMA) to identify trend direction
# - Uses 1w Williams %R to identify overbought/oversold conditions in ranging markets
# - Enters long when price is above 1d KAMA and 1w Williams %R < -80 (oversold)
# - Enters short when price is below 1d KAMA and 1w Williams %R > -20 (overbought)
# - Exits when price crosses back below/above 1d KAMA or Williams %R returns to neutral range (-50 to -50)
# - Designed to capture mean reversion in ranging markets while filtering by trend direction
# - Target: 60-120 total trades over 4 years (15-30/year) with 0.25 position sizing

name = "6h_1dKAMA_1wWilliamsMR"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Get 1w data for Williams %R calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate 1d KAMA (10, 2, 30)
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(close_1d - np.roll(close_1d, 10))
    change[0:10] = change[9]  # Fill first 10 values
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    volatility_rolling = pd.Series(volatility).rolling(window=10, min_periods=10).sum().values
    volatility_rolling[0:10] = volatility_rolling[9]  # Fill first 10 values
    er = change / (volatility_rolling + 1e-10)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate 1w Williams %R (14)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    # Highest high and lowest low over 14 periods
    hh = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    # Williams %R
    williams_r = -100 * (hh - close_1w) / (hh - ll + 1e-10)
    
    # Align 1d KAMA to 6h timeframe
    kama_6h = align_htf_to_ltf(prices, df_1d, kama)
    
    # Align 1w Williams %R to 6h timeframe
    williams_r_6h = align_htf_to_ltf(prices, df_1w, williams_r)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup
        # Skip if any critical value is NaN
        if (np.isnan(kama_6h[i]) or np.isnan(williams_r_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA and Williams %R oversold (< -80)
            if close[i] > kama_6h[i] and williams_r_6h[i] < -80:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and Williams %R overbought (> -20)
            elif close[i] < kama_6h[i] and williams_r_6h[i] > -20:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below KAMA OR Williams %R returns to neutral (> -50)
            if close[i] < kama_6h[i] or williams_r_6h[i] > -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above KAMA OR Williams %R returns to neutral (< -50)
            if close[i] > kama_6h[i] or williams_r_6h[i] < -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals