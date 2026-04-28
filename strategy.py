#!/usr/bin/env python3
"""
1d_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: On daily timeframe, enter long when price breaks above weekly Camarilla R3 level with volume surge and weekly uptrend (price above weekly 200 EMA), short when price breaks below S3 level with volume surge and weekly downtrend. Exit on opposite weekly Camarilla level break. Uses weekly EMA200 trend filter to avoid counter-trend trades. Designed for low trade frequency (~10-25/year) to minimize fee decay in both bull and bear markets. Weekly Camarilla levels provide strong support/resistance based on prior week's range, working well in trending conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 200:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from previous week
    # (H, L, C from previous week)
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Weekly Camarilla levels
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Weekly EMA200 for trend filter
    ema_200 = pd.Series(df_1w['close']).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align all weekly data to daily timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    ema_200_aligned = align_htf_to_ltf(prices, df_1w, ema_200)
    
    # Trend: bullish when price > weekly EMA200, bearish when price < weekly EMA200
    w1_uptrend = close > ema_200_aligned
    w1_downtrend = close < ema_200_aligned
    
    # Volume confirmation: current volume > 2.0x 24-period average (approx 6 days)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_surge = volume > (vol_ma_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for sufficient warmup (weekly EMA200)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_200_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with weekly EMA200 trend alignment and volume surge
        long_entry = close[i] > R3_aligned[i] and w1_uptrend[i] and volume_surge[i]
        short_entry = close[i] < S3_aligned[i] and w1_downtrend[i] and volume_surge[i]
        
        # Exit on opposite weekly Camarilla level break with volume surge
        long_exit = close[i] < S3_aligned[i] and volume_surge[i]
        short_exit = close[i] > R3_aligned[i] and volume_surge[i]
        
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

name = "1d_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "1d"
leverage = 1.0