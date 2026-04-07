#!/usr/bin/env python3
"""
6h_camarilla_pivot_12h_ema_volume_v1
Hypothesis: Camarilla pivot levels on 12h timeframe provide high-probability reversal/fade zones,
while 12h EMA trend filter ensures we trade with the higher timeframe trend. Volume confirmation
reduces false signals. Designed for 6h timeframe to target 15-30 trades/year, minimizing fee drag.
Works in both bull and bear markets by following the 12h trend and using Camarilla levels for
mean reversion in ranging conditions or breakout confirmation in trending markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_12h_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivots and EMA
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate typical price for 12h
    typical_price = (df_12h['high'] + df_12h['low'] + df_12h['close']) / 3
    
    # Calculate Camarilla pivot levels (using previous day's typical price)
    # For each 12h bar, use the previous 12h bar's high, low, close
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla levels: based on previous period's range
    camarilla_S1 = np.zeros(len(df_12h))
    camarilla_S2 = np.zeros(len(df_12h))
    camarilla_S3 = np.zeros(len(df_12h))
    camarilla_S4 = np.zeros(len(df_12h))
    camarilla_R1 = np.zeros(len(df_12h))
    camarilla_R2 = np.zeros(len(df_12h))
    camarilla_R3 = np.zeros(len(df_12h))
    camarilla_R4 = np.zeros(len(df_12h))
    
    for i in range(1, len(df_12h)):
        # Previous period's high, low, close
        ph = high_12h[i-1]
        pl = low_12h[i-1]
        pc = close_12h[i-1]
        
        range_val = ph - pl
        if range_val > 0:
            camarilla_S1[i] = pc - (range_val * 1.0833 / 6)
            camarilla_S2[i] = pc - (range_val * 1.0833 / 4)
            camarilla_S3[i] = pc - (range_val * 1.0833 / 3)
            camarilla_S4[i] = pc - (range_val * 1.0833 / 2)
            camarilla_R1[i] = pc + (range_val * 1.0833 / 6)
            camarilla_R2[i] = pc + (range_val * 1.0833 / 4)
            camarilla_R3[i] = pc + (range_val * 1.0833 / 3)
            camarilla_R4[i] = pc + (range_val * 1.0833 / 2)
        else:
            camarilla_S1[i] = camarilla_S2[i] = camarilla_S3[i] = camarilla_S4[i] = pc
            camarilla_R1[i] = camarilla_R2[i] = camarilla_R3[i] = camarilla_R4[i] = pc
    
    # Align Camarilla levels to 6h timeframe
    S1_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_S1)
    S2_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_S2)
    S3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_S3)
    S4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_S4)
    R1_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_R1)
    R2_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_R2)
    R3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_R3)
    R4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_R4)
    
    # 12h EMA for trend filter
    ema_12h = pd.Series(close_12h).ewm(span=20, min_periods=20).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume confirmation: volume > 20-period average on 6h
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(S1_12h_aligned[i]) or np.isnan(R1_12h_aligned[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: price reaches R3 or breaks below S1
            if close[i] >= R3_12h_aligned[i] or close[i] <= S1_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price reaches S3 or breaks above R1
            if close[i] <= S3_12h_aligned[i] or close[i] >= R1_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Fade at extreme levels (S3/R3) when price is outside normal range
            # Long: price touches or goes below S3, volume confirmed, and above 12h EMA (bullish bias)
            if close[i] <= S3_12h_aligned[i] and volume[i] > vol_ma[i] and close[i] > ema_12h_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price touches or goes above R3, volume confirmed, and below 12h EMA (bearish bias)
            elif close[i] >= R3_12h_aligned[i] and volume[i] > vol_ma[i] and close[i] < ema_12h_aligned[i]:
                position = -1
                signals[i] = -0.25
            # Breakout continuation at S4/R4 with volume
            # Long: price breaks above R4 with volume (bullish breakout)
            elif close[i] >= R4_12h_aligned[i] and volume[i] > vol_ma[i] and close[i] > ema_12h_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Short: price breaks below S4 with volume (bearish breakout)
            elif close[i] <= S4_12h_aligned[i] and volume[i] > vol_ma[i] and close[i] < ema_12h_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals