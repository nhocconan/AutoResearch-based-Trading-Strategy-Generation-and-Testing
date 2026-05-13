#!/usr/bin/env python3
"""
12h_Camarilla_R3_S3_Breakout_1dTrend_Volume
Hypothesis: Use 12h Camarilla R3/S3 breakouts with 1d trend filter (price > EMA50) and volume spike (volume > 1.5x 20-period average) for entries. Exit on opposite Camarilla level (S1/R1) touch or trend reversal. Camarilla levels provide intraday support/resistance; breakouts with volume capture momentum. Trend filter avoids counter-trend trades. Works in bull (catching breakouts) and bear (selling breakdowns) markets. Designed for 12h timeframe to limit trades (12-37/year) and avoid fee drag.
"""

name = "12h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Get daily data for Camarilla calculation (using previous day's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from previous day's OHLC
    # H = high, L = low, C = close of previous day
    H = df_1d['high'].values
    L = df_1d['low'].values
    C = df_1d['close'].values
    
    # Camarilla levels: R4 = C + (H-L)*1.1/2, R3 = C + (H-L)*1.1/4, etc.
    # We need R3, S3, R1, S1
    R3 = C + (H - L) * 1.1 / 4
    S3 = C - (H - L) * 1.1 / 4
    R1 = C + (H - L) * 1.1 / 12
    S1 = C - (H - L) * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe (no extra delay - levels based on closed daily bar)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # Get daily EMA50 for trend filter
    ema_50_1d = pd.Series(C).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate volume average (20-period) for volume spike filter
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if any required data is NaN
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume spike condition: current volume > 1.5x 20-period average
        vol_spike = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # LONG: Price breaks above R3 with volume spike and price above daily EMA50
            if close[i] > R3_aligned[i] and vol_spike and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below S3 with volume spike and price below daily EMA50
            elif close[i] < S3_aligned[i] and vol_spike and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price touches S1 (support) or trend fails (price < EMA50)
            if close[i] <= S1_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price touches R1 (resistance) or trend fails (price > EMA50)
            if close[i] >= R1_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals