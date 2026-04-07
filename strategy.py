#!/usr/bin/env python3
"""
12h_camarilla_pivot_1d_trend_volume_v1
Hypothesis: Daily Camarilla pivot levels act as strong support/resistance. 
In trending markets (price above/below 200 EMA), price often reverses from 
S3/S4 or R3/R4 levels. Volume confirms institutional interest at these levels.
Works in both bull and bear markets by fading extremes with trend filter.
Target: 15-35 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1d_trend_volume_v1"
timezone = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivots and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels for previous day
    # Formula: Based on previous day's high, low, close
    # Resistance levels: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # Support levels: S4 = C - (H-L)*1.1/2, S3 = C - (H-L)*1.1/4, etc.
    # Actually: R4 = Close + (High-Low)*1.1/2, R3 = Close + (High-Low)*1.1/4
    # R2 = Close + (High-Low)*1.1/6, R1 = Close + (High-Low)*1.1/12
    # S1 = Close - (High-Low)*1.1/12, S2 = Close - (High-Low)*1.1/6
    # S3 = Close - (High-Low)*1.1/4, S4 = Close - (High-Low)*1.1/2
    
    prev_high = df_1d['high'].shift(1)
    prev_low = df_1d['low'].shift(1)
    prev_close = df_1d['close'].shift(1)
    
    # Calculate pivot levels
    high_low_range = prev_high - prev_low
    r4 = prev_close + high_low_range * 1.1 / 2
    r3 = prev_close + high_low_range * 1.1 / 4
    r2 = prev_close + high_low_range * 1.1 / 6
    r1 = prev_close + high_low_range * 1.1 / 12
    s1 = prev_close - high_low_range * 1.1 / 12
    s2 = prev_close - high_low_range * 1.1 / 6
    s3 = prev_close - high_low_range * 1.1 / 4
    s4 = prev_close - high_low_range * 1.1 / 2
    
    # Daily 200 EMA for trend filter
    ema_200 = df_1d['close'].ewm(span=200, adjust=False).mean()
    
    # Align all daily data to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200.values)
    
    # Volume confirmation (24-period average = 12 days on 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(24, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_200_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches R3 (take profit) or trend turns bearish
            if close[i] >= r3_aligned[i] or close[i] < ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches S3 (take profit) or trend turns bullish
            if close[i] <= s3_aligned[i] or close[i] > ema_200_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches or goes below S4 with volume and bullish trend
            if (close[i] <= s4_aligned[i] and vol_confirm and 
                close[i] > ema_200_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches or goes above R4 with volume and bearish trend
            elif (close[i] >= r4_aligned[i] and vol_confirm and 
                  close[i] < ema_200_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals