#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_trend_volume_v1
Hypothesis: Camarilla pivot levels from daily timeframe combined with 4h trend filter and volume confirmation.
In trending markets, price tends to respect daily pivot levels (S3/S4 for shorts, R3/R4 for longs).
Uses daily pivot levels as institutional support/resistance, 4h EMA for trend filter, and volume spike for confirmation.
Designed for 4h timeframe to capture multi-day moves with low frequency (target: 20-50 trades/year) to minimize fee drag.
Works in both bull and bear markets by following the trend defined by higher timeframes.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_trend_volume_v1"
timeframe = "4h"
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
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla levels (based on previous day)
    # Formula: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, R2 = C + (H-L)*1.1/6, R1 = C + (H-L)*1.1/12
    # S1 = C - (H-L)*1.1/12, S2 = C - (H-L)*1.1/6, S3 = C - (H-L)*1.1/4, S4 = C - (H-L)*1.1/2
    daily_close = df_1d['close'].shift(1).values
    daily_high = df_1d['high'].shift(1).values
    daily_low = df_1d['low'].shift(1).values
    daily_range = daily_high - daily_low
    
    # Camarilla levels
    r4 = daily_close + daily_range * 1.1 / 2
    r3 = daily_close + daily_range * 1.1 / 4
    r2 = daily_close + daily_range * 1.1 / 6
    r1 = daily_close + daily_range * 1.1 / 12
    s1 = daily_close - daily_range * 1.1 / 12
    s2 = daily_close - daily_range * 1.1 / 6
    s3 = daily_close - daily_range * 1.1 / 4
    s4 = daily_close - daily_range * 1.1 / 2
    
    # 4h EMA for trend filter
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align all daily data to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Volume confirmation (6-period average = 1 day)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):
        # Skip if required data not available
        if (np.isnan(r4_aligned[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price crosses below R3 or trend turns bearish
            if close[i] <= r3_aligned[i] or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price crosses above S3 or trend turns bullish
            if close[i] >= s3_aligned[i] or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches or breaks above R4 with volume and bullish trend
            if (close[i] >= r4_aligned[i] and vol_confirm and 
                close[i] > ema_20_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches or breaks below S4 with volume and bearish trend
            elif (close[i] <= s4_aligned[i] and vol_confirm and 
                  close[i] < ema_20_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals