#!/usr/bin/env python3
# 6h_Stochastic_14_3_3_Bollinger_20_2_Trend_1dEMA50
# Hypothesis: Stochastic oscillator (14,3,3) identifies overbought/oversold conditions, 
# Bollinger Bands (20,2) provide volatility context and dynamic support/resistance,
# 1-day EMA50 establishes trend direction. Long when Stochastic crosses above 20 from below 
# in uptrend near lower Bollinger Band; short when Stochastic crosses below 80 from above 
# in downtrend near upper Bollinger Band. Designed for low-frequency, high-conviction trades 
# in both bull and bear markets by combining mean reversion with trend filter.

name = "6h_Stochastic_14_3_3_Bollinger_20_2_Trend_1dEMA50"
timeframe = "6h"
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

    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)

    # Calculate Bollinger Bands (20,2) on 6h data
    close_series = pd.Series(close)
    bb_middle = close_series.rolling(window=20, min_periods=20).mean().values
    bb_std = close_series.rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std

    # Calculate Stochastic Oscillator (14,3,3) on 6h data
    # %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    k_percent = 100 * (close - lowest_low) / (highest_high - lowest_low)
    # Handle division by zero when high == low
    k_percent = np.where((highest_high - lowest_low) == 0, 50, k_percent)
    
    # %D = 3-period SMA of %K
    k_series = pd.Series(k_percent)
    d_percent = k_series.rolling(window=3, min_periods=3).mean().values
    
    # Slow %K = 3-period SMA of %K (same as %D in standard settings)
    slow_k = d_percent.copy()
    # Slow %D = 3-period SMA of slow %K
    slow_d = pd.Series(slow_k).rolling(window=3, min_periods=3).mean().values

    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short

    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(bb_upper[i]) or 
            np.isnan(bb_lower[i]) or np.isnan(slow_k[i]) or np.isnan(slow_d[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue

        if position == 0:
            # LONG: Stochastic crosses above 20 from below in uptrend near lower BB
            if (slow_k[i] > 20 and slow_k[i-1] <= 20 and 
                slow_d[i] > slow_d[i-1] and  # Stochastic momentum up
                close[i] > ema50_1d_aligned[i] and  # Uptrend filter
                close[i] <= bb_lower[i] * 1.02):  # Near lower Bollinger Band (within 2%)
                signals[i] = 0.25
                position = 1
            # SHORT: Stochastic crosses below 80 from above in downtrend near upper BB
            elif (slow_k[i] < 80 and slow_k[i-1] >= 80 and 
                  slow_d[i] < slow_d[i-1] and  # Stochastic momentum down
                  close[i] < ema50_1d_aligned[i] and  # Downtrend filter
                  close[i] >= bb_upper[i] * 0.98):  # Near upper Bollinger Band (within 2%)
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Stochastic crosses above 80 (overbought) or trend change
            if slow_k[i] > 80 and slow_k[i-1] <= 80:
                signals[i] = 0.0
                position = 0
            elif close[i] < ema50_1d_aligned[i]:  # Trend breakdown
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Stochastic crosses below 20 (oversold) or trend change
            if slow_k[i] < 20 and slow_k[i-1] >= 20:
                signals[i] = 0.0
                position = 0
            elif close[i] > ema50_1d_aligned[i]:  # Trend reversal
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25

    return signals