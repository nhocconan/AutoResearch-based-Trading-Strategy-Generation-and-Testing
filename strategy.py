#!/usr/bin/env python3
"""
4h_Williams_Alligator_Gap_Close
Long when price closes above Alligator Jaw (TEETH) in uptrend with volume confirmation.
Short when price closes below Alligator Jaw (TEETH) in downtrend with volume confirmation.
Exit when price crosses back through Alligator Jaw.
Uses Williams Alligator (13,8,5 SMAs shifted) for trend definition and gap filtering.
Targets 20-40 trades per year with strict entry conditions.
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Williams Alligator on 4h: Jaw (13), Teeth (8), Lips (5) - all SMAs with shift
    jaw_period = 13
    teeth_period = 8
    lips_period = 5
    jaw_shift = 8
    teeth_shift = 5
    lips_shift = 3
    
    # Jaw (13-period SMA, shifted 8 bars forward)
    jaw = np.full(n, np.nan)
    for i in range(jaw_period - 1 + jaw_shift, n):
        jaw[i] = np.mean(close[i - jaw_shift - jaw_period + 1:i - jaw_shift + 1])
    
    # Teeth (8-period SMA, shifted 5 bars forward)
    teeth = np.full(n, np.nan)
    for i in range(teeth_period - 1 + teeth_shift, n):
        teeth[i] = np.mean(close[i - teeth_shift - teeth_period + 1:i - teeth_shift + 1])
    
    # Lips (5-period SMA, shifted 3 bars forward)
    lips = np.full(n, np.nan)
    for i in range(lips_period - 1 + lips_shift, n):
        lips[i] = np.mean(close[i - lips_shift - lips_period + 1:i - lips_shift + 1])
    
    # Align 1d EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need all Alligator components, EMA1d, and volume MA20
    start_idx = max(jaw_period - 1 + jaw_shift, teeth_period - 1 + teeth_shift, 
                    lips_period - 1 + lips_shift, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(jaw[i]) or np.isnan(teeth[i]) or np.isnan(lips[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.3 * vol_avg
        
        # Trend filter: price relative to 1d EMA50
        uptrend = price > ema_1d_aligned[i]
        downtrend = price < ema_1d_aligned[i]
        
        if position == 0:
            # Long: price closes above Teeth in uptrend with volume confirmation
            if (price > teeth[i] and 
                close[i - 1] <= teeth[i - 1] and  # crossed above teeth on close
                uptrend and vol_filter):
                signals[i] = size
                position = 1
            # Short: price closes below Teeth in downtrend with volume confirmation
            elif (price < teeth[i] and 
                  close[i - 1] >= teeth[i - 1] and  # crossed below teeth on close
                  downtrend and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below Jaw (stronger signal)
            if price < jaw[i] and close[i - 1] >= jaw[i - 1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above Jaw (stronger signal)
            if price > jaw[i] and close[i - 1] <= jaw[i - 1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Williams_Alligator_Gap_Close"
timeframe = "4h"
leverage = 1.0