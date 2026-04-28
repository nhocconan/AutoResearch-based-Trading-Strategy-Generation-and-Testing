#!/usr/bin/env python3
"""
4h_Camarilla_R3S3_Breakout_1dTrend_Volume
Hypothesis: On 4h timeframe, enter long when price breaks above Camarilla R3 with volume surge and 1d uptrend (close > EMA34), short when price breaks below S3 with volume surge and 1d downtrend. Exit on opposite breakout with volume. Uses volume confirmation and 1d trend filter to reduce false signals. Designed for moderate trade frequency (~25-40/year) to balance signal quality and fee drag in both bull and bear markets.
"""

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate daily 34 EMA for trend filter
    close_1d = df_1d['close'].values
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Daily trend: bullish when close > EMA34
    daily_uptrend = close_1d > ema34_1d
    daily_downtrend = close_1d < ema34_1d
    daily_uptrend_aligned = align_htf_to_ltf(prices, df_1d, daily_uptrend.astype(float))
    daily_downtrend_aligned = align_htf_to_ltf(prices, df_1d, daily_downtrend.astype(float))
    
    # Calculate Camarilla levels from previous day
    # Typical price = (high + low + close) / 3
    typical_price = (high + low + close) / 3.0
    # Use previous day's typical price (shift by 1)
    prev_typical = pd.Series(typical_price).shift(1).values
    
    # Calculate range (high - low) of previous day
    prev_high = pd.Series(high).shift(1).values
    prev_low = pd.Series(low).shift(1).values
    prev_range = prev_high - prev_low
    
    # Camarilla levels
    R3 = prev_typical + (prev_range * 1.1 / 4)
    S3 = prev_typical - (prev_range * 1.1 / 4)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(daily_uptrend_aligned[i]) or
            np.isnan(daily_downtrend_aligned[i]) or np.isnan(R3[i]) or np.isnan(S3[i]) or
            np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with daily trend alignment and volume surge
        long_entry = close[i] > R3[i] and daily_uptrend_aligned[i] > 0.5 and volume_surge[i]
        short_entry = close[i] < S3[i] and daily_downtrend_aligned[i] > 0.5 and volume_surge[i]
        
        # Exit on opposite Camarilla level break with volume surge
        long_exit = close[i] < S3[i] and volume_surge[i]
        short_exit = close[i] > R3[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.25  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.25   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0