#!/usr/bin/env python3
"""
4h Camarilla Pivot Reversal with Volume Spike and Trend Filter
Hypothesis: Price often reverses from key intraday support/resistance levels (Camarilla levels)
calculated from the previous day's range. Entries are taken at S3/R3 levels with volume
confirmation and trend alignment (using 1-day EMA) to avoid counter-trend trades.
Designed for 20-50 trades/year on 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot calculation (once before loop)
    df_d = get_htf_data(prices, '1d')
    
    # Previous day's high, low, close
    prev_high = df_d['high'].shift(1).values
    prev_low = df_d['low'].shift(1).values
    prev_close = df_d['close'].shift(1).values
    
    # Calculate Camarilla levels: S3, S2, S1, R1, R2, R3
    # Range = previous day's high - low
    range_val = prev_high - prev_low
    
    # Camarilla formulas
    s3 = prev_close - (range_val * 1.1 / 6)
    s2 = prev_close - (range_val * 1.1 / 4)
    s1 = prev_close - (range_val * 1.1 / 2)
    r1 = prev_close + (range_val * 1.1 / 2)
    r2 = prev_close + (range_val * 1.1 / 4)
    r3 = prev_close + (range_val * 1.1 / 6)
    
    # Align Camarilla levels to 4h timeframe
    s3_aligned = align_htf_to_ltf(prices, df_d, s3)
    s2_aligned = align_htf_to_ltf(prices, df_d, s2)
    s1_aligned = align_htf_to_ltf(prices, df_d, s1)
    r1_aligned = align_htf_to_ltf(prices, df_d, r1)
    r2_aligned = align_htf_to_ltf(prices, df_d, r2)
    r3_aligned = align_htf_to_ltf(prices, df_d, r3)
    
    # Volume spike: 2x 20-period average on 4h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Trend filter: 1-day EMA (34) aligned to 4h
    ema_34_d = pd.Series(df_d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_d, ema_34_d)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        if (np.isnan(s3_aligned[i]) or np.isnan(s2_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(r1_aligned[i]) or np.isnan(r2_aligned[i]) or np.isnan(r3_aligned[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema_34_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema = ema_34_aligned[i]
        
        if position == 0:
            # Long: price at or below S3 with volume spike and uptrend (price > EMA)
            if price <= s3_aligned[i] and volume_spike[i] and price > ema:
                signals[i] = 0.25
                position = 1
            # Short: price at or above R3 with volume spike and downtrend (price < EMA)
            elif price >= r3_aligned[i] and volume_spike[i] and price < ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position
            signals[i] = 0.25
            # Exit: price reaches S1 or trend turns bearish
            if price >= s1_aligned[i] or price < ema:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position
            signals[i] = -0.25
            # Exit: price reaches R1 or trend turns bullish
            if price <= r1_aligned[i] or price > ema:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Camarilla_Pivot_Reversal_Volume_Trend"
timeframe = "4h"
leverage = 1.0