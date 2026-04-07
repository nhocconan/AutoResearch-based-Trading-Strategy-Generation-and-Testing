#!/usr/bin/env python3
"""
4h_camarilla_pivot_1d_trend_volume_v2
Hypothesis: Camarilla pivot levels on 1d timeframe identify key support/resistance levels, 
combined with 1d EMA trend filter and volume confirmation. Works in both bull and bear markets 
by buying near support in uptrends and selling near resistance in downtrends. Target: 20-40 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_camarilla_pivot_1d_trend_volume_v2"
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
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels (based on previous day's range)
    # Typical price = (H + L + C) / 3
    typical_price = (df_1d['high'] + df_1d['low'] + df_1d['close']) / 3
    # Range = High - Low
    daily_range = df_1d['high'] - df_1d['low']
    
    # Camarilla levels (using previous day's data)
    # S1 = Close - (Range * 1.1/12)
    # S2 = Close - (Range * 1.1/6)
    # S3 = Close - (Range * 1.1/4)
    # S4 = Close - (Range * 1.1/2)
    # R1 = Close + (Range * 1.1/12)
    # R2 = Close + (Range * 1.1/6)
    # R3 = Close + (Range * 1.1/4)
    # R4 = Close + (Range * 1.1/2)
    
    prev_close = df_1d['close'].shift(1)
    prev_range = df_1d['high'].shift(1) - df_1d['low'].shift(1)
    
    # Calculate levels
    s1 = prev_close - (prev_range * 1.1 / 12)
    s2 = prev_close - (prev_range * 1.1 / 6)
    s3 = prev_close - (prev_range * 1.1 / 4)
    s4 = prev_close - (prev_range * 1.1 / 2)
    r1 = prev_close + (prev_range * 1.1 / 12)
    r2 = prev_close + (prev_range * 1.1 / 6)
    r3 = prev_close + (prev_range * 1.1 / 4)
    r4 = prev_close + (prev_range * 1.1 / 2)
    
    # Daily EMA for trend filter (20-period)
    ema_20 = df_1d['close'].ewm(span=20, adjust=False).mean()
    
    # Align all daily data to 4h timeframe
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1.values)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2.values)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3.values)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4.values)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1.values)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2.values)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3.values)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4.values)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20.values)
    
    # Volume confirmation (20-period average = ~3.3 days on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(s1_aligned[i]) or np.isnan(r1_aligned[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x average volume
        vol_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price reaches R3 or trend turns bearish
            if close[i] >= r3_aligned[i] or close[i] < ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: price reaches S3 or trend turns bullish
            if close[i] <= s3_aligned[i] or close[i] > ema_20_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price touches S1 with volume and bullish trend
            if (close[i] <= s1_aligned[i] and vol_confirm and 
                close[i] > ema_20_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price touches R1 with volume and bearish trend
            elif (close[i] >= r1_aligned[i] and vol_confirm and 
                  close[i] < ema_20_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals