#!/usr/bin/env python3
"""
4h Camarilla R1/S1 Breakout with Volume Spike and 1d EMA Trend Filter
Long: Price breaks above R1 + volume > 2x 4h volume SMA(20) + price > 1d EMA(50)
Short: Price breaks below S1 + volume > 2x 4h volume SMA(20) + price < 1d EMA(50)
Exit: Price retests the pivot point (PP) or opposite stop via ATR
Uses Camarilla levels from daily pivot, volume confirmation, and trend filter
Target: 20-30 trades/year per symbol (80-120 total over 4 years)
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
    
    # Get 1d data for Camarilla pivot and EMA trend
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate typical price for pivot
    # For each bar, we need previous day's HLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Shift by 1 to get previous day's values for current bar calculation
    pp_vals = (np.roll(high_1d, 1) + np.roll(low_1d, 1) + np.roll(close_1d, 1)) / 3
    pp_vals[0] = np.nan  # first bar has no previous day
    
    # Calculate Camarilla levels: R1 = PP + 1.1*(H-L)/12, S1 = PP - 1.1*(H-L)/12
    hl_range = np.roll(high_1d, 1) - np.roll(low_1d, 1)
    r1 = pp_vals + 1.1 * hl_range / 12
    s1 = pp_vals - 1.1 * hl_range / 12
    
    # Align to 4h
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp_vals)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Calculate 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 4h volume SMA(20) for volume filter
    vol_sma_4h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(30, 50)  # need EMA50 and volume SMA
    
    for i in range(start_idx, n):
        if (np.isnan(pp_aligned[i]) or np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_4h[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_sma_val = vol_sma_4h[i]
        ema_50_val = ema_50_1d_aligned[i]
        pp_val = pp_aligned[i]
        r1_val = r1_aligned[i]
        s1_val = s1_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R1 + volume > 2x SMA + price > 1d EMA50
            if price > r1_val and close[i-1] <= r1_val and vol > 2.0 * vol_sma_val and price > ema_50_val:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S1 + volume > 2x SMA + price < 1d EMA50
            elif price < s1_val and close[i-1] >= s1_val and vol > 2.0 * vol_sma_val and price < ema_50_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price retests PP or breaks below S1
            if price < pp_val or price < s1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price retests PP or breaks above R1
            if price > pp_val or price > r1_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R1S1_Breakout_VolumeSpike_1dEMA50"
timeframe = "4h"
leverage = 1.0