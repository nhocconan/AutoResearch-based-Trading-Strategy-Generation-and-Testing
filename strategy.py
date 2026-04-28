#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: On 4-hour timeframe, enter long when price breaks above Camarilla R3 level with volume surge and 1d uptrend (price above 100 EMA), short when price breaks below S3 level with volume surge and 1d downtrend. Exit on opposite Camarilla level break. Uses 1d EMA100 trend filter to avoid counter-trend trades. Designed for low trade frequency (~20-40/year) to minimize fee decay.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day
    # (H, L, C from previous day)
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Camarilla levels - focusing on R3 and S3 for breakouts
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # 1d EMA100 for trend filter
    ema_100 = pd.Series(df_1d['close']).ewm(span=100, adjust=False, min_periods=100).mean().values
    
    # Align all 1d data to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    ema_100_aligned = align_htf_to_ltf(prices, df_1d, ema_100)
    
    # Trend: bullish when price > EMA100, bearish when price < EMA100
    d1_uptrend = close > ema_100_aligned
    d1_downtrend = close < ema_100_aligned
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_surge = volume > (vol_ma_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_100_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with 1d EMA100 trend alignment and volume surge
        long_entry = close[i] > R3_aligned[i] and d1_uptrend[i] and volume_surge[i]
        short_entry = close[i] < S3_aligned[i] and d1_downtrend[i] and volume_surge[i]
        
        # Exit on opposite Camarilla level break with volume surge
        long_exit = close[i] < S3_aligned[i] and volume_surge[i]
        short_exit = close[i] > R3_aligned[i] and volume_surge[i]
        
        if long_entry and position <= 0:
            signals[i] = 0.30
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.30
            position = -1
        elif long_exit and position == 1:
            signals[i] = -0.30  # Reverse to short
            position = -1
        elif short_exit and position == -1:
            signals[i] = 0.30   # Reverse to long
            position = 1
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.30
            elif position == -1:
                signals[i] = -0.30
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0