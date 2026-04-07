#!/usr/bin/env python3
"""
12h_camarilla_pivot_1w_trend_volume_v1
Hypothesis: Camarilla pivot levels from 1-week combined with 1-day trend filter and volume confirmation.
Trades long when price touches S3/S4 support in bullish weekly trend, short when touching R3/R4 resistance in bearish weekly trend.
Works in both bull and bear markets by aligning with weekly trend direction.
Target: 12-37 trades/year (50-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_1w_trend_volume_v1"
timeframe = "12h"
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
    
    # Weekly data for Camarilla pivot levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate weekly Camarilla pivot levels
    # Based on previous week's high, low, close
    ph = df_1w['high'].shift(1).values  # previous week high
    pl = df_1w['low'].shift(1).values   # previous week low
    pc = df_1w['close'].shift(1).values # previous week close
    
    # Pivot point
    pp = (ph + pl + pc) / 3.0
    # Camarilla levels
    r4 = pp + (ph - pl) * 1.1 / 2.0
    r3 = pp + (ph - pl) * 1.1 / 4.0
    s3 = pp - (ph - pl) * 1.1 / 4.0
    s4 = pp - (ph - pl) * 1.1 / 2.0
    
    # Daily EMA for trend filter (20-period)
    ema_20 = df_1d['close'].ewm(span=20, adjust=False).mean().values
    
    # Align all weekly and daily data to 12h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1w, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1w, s4)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i]) or
            vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below S3 or weekly trend turns bearish
            if close[i] < s3_aligned[i] or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price closes above R3 or weekly trend turns bullish
            if close[i] > r3_aligned[i] or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches S3/S4 support with volume and bullish weekly trend
            if vol_confirm and close[i] > ema_20_aligned[i]:
                # Check if price is near S3 or S4 (within 0.2% of level)
                near_s3 = abs(close[i] - s3_aligned[i]) / s3_aligned[i] < 0.002
                near_s4 = abs(close[i] - s4_aligned[i]) / s4_aligned[i] < 0.002
                if near_s3 or near_s4:
                    position = 1
                    signals[i] = 0.25
            # Short entry: price touches R3/R4 resistance with volume and bearish weekly trend
            elif vol_confirm and close[i] < ema_20_aligned[i]:
                # Check if price is near R3 or R4 (within 0.2% of level)
                near_r3 = abs(close[i] - r3_aligned[i]) / r3_aligned[i] < 0.002
                near_r4 = abs(close[i] - r4_aligned[i]) / r4_aligned[i] < 0.002
                if near_r3 or near_r4:
                    position = -1
                    signals[i] = -0.25
    
    return signals