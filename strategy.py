#!/usr/bin/env python3
"""
6h_camarilla_pivot_1d_ema_volume_v1
Hypothesis: Camarilla pivot levels from 1-day timeframe combined with EMA trend filter and volume confirmation on 6h chart.
In long: price rebounds from S3/S4 levels with volume confirmation and price above 12h EMA50.
In short: price rebounds from R3/R4 levels with volume confirmation and price below 12h EMA50.
Uses Camarilla's institutional reversal levels, EMA for trend direction, and volume for confirmation.
Designed for 15-30 trades/year on 6h timeframe with clear reversal logic that works in both bull and bear markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_1d_ema_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels from previous day
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's OHLC for Camarilla calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    prev_high[0] = high_1d[0]  # First day uses same day
    prev_low[0] = low_1d[0]
    prev_close[0] = close_1d[0]
    
    # Camarilla calculations
    range_1d = prev_high - prev_low
    camarilla_S3 = prev_close - (range_1d * 1.1 / 6)
    camarilla_S4 = prev_close - (range_1d * 1.1 / 4)
    camarilla_R3 = prev_close + (range_1d * 1.1 / 6)
    camarilla_R4 = prev_close + (range_1d * 1.1 / 4)
    
    # Align Camarilla levels to 6h timeframe
    S3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S3)
    S4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S4)
    R3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R4)
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    ema50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if data not available
        if (np.isnan(S3_aligned[i]) or np.isnan(S4_aligned[i]) or 
            np.isnan(R3_aligned[i]) or np.isnan(R4_aligned[i]) or
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema50_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above average
        vol_confirmed = volume[i] > vol_ma[i]
        
        # Price relative to Camarilla levels
        near_S3_S4 = (close[i] <= S3_aligned[i] * 1.002) or (close[i] <= S4_aligned[i] * 1.002)
        near_R3_R4 = (close[i] >= R3_aligned[i] * 0.998) or (close[i] >= R4_aligned[i] * 0.998)
        
        # 12h trend filter
        above_12h_ema50 = close[i] > ema50_12h_aligned[i]
        below_12h_ema50 = close[i] < ema50_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit: price moves below S4 or trend turns bearish
            if close[i] < S4_aligned[i] or below_12h_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price moves above R4 or trend turns bullish
            if close[i] > R4_aligned[i] or above_12h_ema50:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long: price near S3/S4 with volume confirmation and bullish trend
            if near_S3_S4 and vol_confirmed and above_12h_ema50:
                position = 1
                signals[i] = 0.25
            # Short: price near R3/R4 with volume confirmation and bearish trend
            elif near_R3_R4 and vol_confirmed and below_12h_ema50:
                position = -1
                signals[i] = -0.25
    
    return signals