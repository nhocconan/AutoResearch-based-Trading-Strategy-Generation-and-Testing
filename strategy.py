#!/usr/bin/env python3
"""
12h_1d_Wick_Reversal_v1
Hypothesis: Price rejection at key daily levels (wick testing) with volume confirmation.
Long when price tests daily support (low) with long lower wick and volume > avg,
short when tests daily resistance (high) with long upper wick and volume > avg.
Uses 1d support/resistance levels for institutional reference. Works in both bull/bear
as it captures rejection at key levels. Target: 50-150 trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_1d_Wick_Reversal_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for support/resistance
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high/low for S/R
    prev_high = df_1d['high'].iloc[-2] if len(df_1d) >= 2 else df_1d['high'].iloc[-1]
    prev_low = df_1d['low'].iloc[-2] if len(df_1d) >= 2 else df_1d['low'].iloc[-1]
    
    # Align daily S/R to 12h timeframe
    daily_high_array = np.full(len(df_1d), prev_high)
    daily_low_array = np.full(len(df_1d), prev_low)
    daily_high_aligned = align_htf_to_ltf(prices, df_1d, daily_high_array)
    daily_low_aligned = align_htf_to_ltf(prices, df_1d, daily_low_array)
    
    # Volume confirmation: current volume > 1.3x 20-period average
    volume_series = pd.Series(volume)
    vol_ma = volume_series.rolling(window=20, min_periods=20).mean()
    vol_ratio = volume_series / vol_ma
    vol_ratio = vol_ratio.fillna(1.0).values
    
    # Wick calculations
    body_size = np.abs(close - open_prices) if 'open' in prices else np.abs(close - np.roll(close, 1))
    # For first bar, use close-open approximation
    open_prices = prices['open'].values
    body_size = np.abs(close - open_prices)
    upper_wick = high - np.maximum(close, open_prices)
    lower_wick = np.minimum(close, open_prices) - low
    
    # Avoid division by zero
    body_size_safe = np.where(body_size == 0, 0.001, body_size)
    upper_wick_ratio = upper_wick / body_size_safe
    lower_wick_ratio = lower_wick / body_size_safe
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if any data invalid
        if (np.isnan(daily_high_aligned[i]) or np.isnan(daily_low_aligned[i]) or
            np.isnan(vol_ratio[i]) or np.isnan(upper_wick_ratio[i]) or 
            np.isnan(lower_wick_ratio[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long reversal: price tests daily support with strong lower wick
        near_support = low[i] <= daily_low_aligned[i] * 1.002  # within 0.2% of support
        strong_lower_wick = lower_wick_ratio[i] > 2.0  # wick at least 2x body
        volume_confirm = vol_ratio[i] > 1.3
        
        # Short reversal: price tests daily resistance with strong upper wick
        near_resistance = high[i] >= daily_high_aligned[i] * 0.998  # within 0.2% of resistance
        strong_upper_wick = upper_wick_ratio[i] > 2.0  # wick at least 2x body
        
        long_signal = near_support and strong_lower_wick and volume_confirm
        short_signal = near_resistance and strong_upper_wick and volume_confirm
        
        # Exit: price moves back toward opposite side of daily range
        daily_mid = (daily_high_aligned[i] + daily_low_aligned[i]) / 2
        long_exit = position == 1 and close[i] < daily_mid
        short_exit = position == -1 and close[i] > daily_mid
        
        # Signal logic
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals