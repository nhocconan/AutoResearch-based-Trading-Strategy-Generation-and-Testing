#!/usr/bin/env python3
"""
4h Camarilla R1/S1 Breakout with Volume Spike and 1d Trend Filter
Long: Price breaks above R1 AND volume > 2x 4h volume SMA(20) AND close > 1d EMA(50)
Short: Price breaks below S1 AND volume > 2x 4h volume SMA(20) AND close < 1d EMA(50)
Exit: Price crosses back below R1 (long) or above S1 (short)
Uses Camarilla levels from daily pivot, volume confirmation, and daily trend filter
Target: 20-40 trades/year per symbol (80-160 total over 4 years)
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily pivot and Camarilla levels
    # Using previous day's OHLC for today's levels
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels: R1 = close + range * 1.1/12, S1 = close - range * 1.1/12
    r1 = prev_close + range_hl * 1.1 / 12.0
    s1 = prev_close - range_hl * 1.1 / 12.0
    
    # Align Camarilla levels to 4h timeframe (they update only at daily open)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h volume SMA(20) for volume filter
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 50)  # Need enough data for EMA50
    
    for i in range(start_idx, n):
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        ema_50_val = ema_50_1d_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R1 + volume > 2x SMA + close > daily EMA50
            if price > r1_val and vol > 2.0 * vol_sma_val and close[i] > ema_50_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + volume > 2x SMA + close < daily EMA50
            elif price < s1_val and vol > 2.0 * vol_sma_val and close[i] < ema_50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses back below R1
            if price < r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses back above S1
            if price > s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_VolumeSpike_1dEMA50"
timeframe = "4h"
leverage = 1.0