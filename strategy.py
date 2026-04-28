#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1wTrend_Volume
Hypothesis: On 12-hour timeframe, enter long when price breaks above weekly Camarilla R3 level with volume surge and weekly uptrend (price above weekly 34 EMA), short when price breaks below weekly S3 level with volume surge and weekly downtrend. Exit on opposite weekly Camarilla level break. Uses weekly EMA34 trend filter to avoid counter-trend trades. Designed for low trade frequency (~15-30/year) to minimize fee decay.
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
    
    # Get weekly data for Camarilla calculation and trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly Camarilla levels from previous week
    prev_high = df_1w['high'].shift(1).values
    prev_low = df_1w['low'].shift(1).values
    prev_close = df_1w['close'].shift(1).values
    
    # Weekly Camarilla levels - focusing on R3 and S3 for breakouts
    R3 = prev_close + (prev_high - prev_low) * 1.1 / 2
    S3 = prev_close - (prev_high - prev_low) * 1.1 / 2
    
    # Weekly EMA34 for trend filter
    ema_34 = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align all weekly data to 12h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1w, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1w, S3)
    ema_34_aligned = align_htf_to_ltf(prices, df_1w, ema_34)
    
    # Trend: bullish when price > EMA34, bearish when price < EMA34
    w_uptrend = close > ema_34_aligned
    w_downtrend = close < ema_34_aligned
    
    # Volume confirmation: current volume > 2.0x 24-period average
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_surge = volume > (vol_ma_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for sufficient warmup
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(volume_surge[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions with weekly EMA34 trend alignment and volume surge
        long_entry = close[i] > R3_aligned[i] and w_uptrend[i] and volume_surge[i]
        short_entry = close[i] < S3_aligned[i] and w_downtrend[i] and volume_surge[i]
        
        # Exit on opposite weekly Camarilla level break with volume surge
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

name = "12h_Camarilla_R3_S3_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0