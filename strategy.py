# 6h_russell_bull_bear_power_v1
# Bull/Bear Power with 1d EMA200 filter and 6m EMA13 pullback for trend continuation.
# Works in bull markets by buying dips in uptrends, works in bear markets by selling rallies in downtrends.
# Uses Bull Power (High - EMA) and Bear Power (Low - EMA) to measure buying/selling pressure behind price moves.
# Target: 15-30 trades per year (60-120 over 4 years) with controlled risk.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_russell_bull_bear_power_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate EMA13 on 6h for pullback entries
    alpha = 2 / (13 + 1)
    ema13 = np.zeros(n)
    ema13[0] = close[0]
    for i in range(1, n):
        ema13[i] = alpha * close[i] + (1 - alpha) * ema13[i-1]
    
    # Get daily data for trend filter and Bull/Bear Power
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 200:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily for trend filter
    alpha_200 = 2 / (200 + 1)
    ema200_1d = np.zeros(len(df_1d))
    ema200_1d[0] = close_1d[0]
    for i in range(1, len(df_1d)):
        ema200_1d[i] = alpha_200 * close_1d[i] + (1 - alpha_200) * ema200_1d[i-1]
    
    # Calculate Bull Power and Bear Power on daily
    bull_power = high_1d - ema200_1d  # Measures buying strength
    bear_power = low_1d - ema200_1d   # Measures selling strength (negative values)
    
    # Align daily indicators to 6h timeframe
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(200, n):
        # Skip if any required data is invalid
        if np.isnan(ema13[i]) or np.isnan(ema200_1d_aligned[i]) or \
           np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: trend turns bearish OR bear power increases (selling pressure)
            if close[i] < ema200_1d_aligned[i] or bear_power_aligned[i] > bear_power_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: trend turns bullish OR bull power increases (buying pressure)
            if close[i] > ema200_1d_aligned[i] or bull_power_aligned[i] > bull_power_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Enter long: bullish trend + bull power expansion + pullback to EMA13
            if (close[i] > ema200_1d_aligned[i] and  # Uptrend filter
                bull_power_aligned[i] > bull_power_aligned[i-1] and  # Increasing buying pressure
                low[i] <= ema13[i] * 1.002 and  # Pullback to EMA13 (within 0.2%)
                close[i] > ema13[i]):  # Close above EMA13 for confirmation
                position = 1
                signals[i] = 0.25
            # Enter short: bearish trend + bear power expansion + pullback to EMA13
            elif (close[i] < ema200_1d_aligned[i] and  # Downtrend filter
                  bear_power_aligned[i] < bear_power_aligned[i-1] and  # Increasing selling pressure (more negative)
                  high[i] >= ema13[i] * 0.998 and  # Pullback to EMA13 (within 0.2%)
                  close[i] < ema13[i]):  # Close below EMA13 for confirmation
                position = -1
                signals[i] = -0.25
    
    return signals