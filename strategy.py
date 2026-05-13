#!/usr/bin/env python3
"""
1d_WickReversal_Volume_Spike
Hypothesis: Long wicks indicate rejection of price levels and potential reversals. Combines long upper/lower wick detection with volume spikes on the 1-day timeframe. Uses 1-week EMA200 as trend filter to align with higher timeframe momentum. Designed for 1d timeframe to capture swing reversals with low trade frequency (~10-25/year), minimizing fee impact while capturing momentum shifts in both bull and bear markets.
"""

name = "1d_WickReversal_Volume_Spike"
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
    open_price = prices['open'].values
    
    # Get 1w data for EMA200 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA200 for trend filter
    ema_200_1w = pd.Series(close_1w).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1w EMA200 to 1d timeframe
    ema_200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_200_1w)
    
    # Calculate body and wick sizes
    body_size = np.abs(close - open_price)
    total_range = high - low
    lower_wick = np.minimum(open_price, close) - low
    upper_wick = high - np.maximum(open_price, close)
    
    # Avoid division by zero
    total_range_safe = np.where(total_range == 0, 1e-10, total_range)
    
    # Long lower wick: lower wick >= 60% of total range
    long_lower_wick = lower_wick >= 0.6 * total_range_safe
    # Long upper wick: upper wick >= 60% of total range
    long_upper_wick = upper_wick >= 0.6 * total_range_safe
    
    # Volume spike: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_200_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Long lower wick + volume spike + price above 1w EMA200 (uptrend filter)
            if long_lower_wick[i] and vol_spike[i] and close[i] > ema_200_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Long upper wick + volume spike + price below 1w EMA200 (downtrend filter)
            elif long_upper_wick[i] and vol_spike[i] and close[i] < ema_200_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 1w EMA200 or long upper wick appears
            if close[i] < ema_200_1w_aligned[i] or long_upper_wick[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 1w EMA200 or long lower wick appears
            if close[i] > ema_200_1w_aligned[i] or long_lower_wick[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals