#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_1wTrend_Volume
Camarilla pivot breakout on 12h with 1w trend filter and volume confirmation.
Long when price breaks above R1 with volume > 1.5x average and 1w trend up.
Short when price breaks below S1 with volume > 1.5x average and 1w trend down.
Exit when price returns to the Camarilla midline (H4/L4 level) or trend fails.
Target: 12-37 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate average volume for confirmation
    vol_ma_period = 20
    vol_ma = np.full(n, np.nan)
    if n >= vol_ma_period:
        vol_sum = np.nansum(volume[:vol_ma_period])
        vol_ma[vol_ma_period - 1] = vol_sum / vol_ma_period
        for i in range(vol_ma_period, n):
            vol_sum = vol_sum - volume[i - vol_ma_period] + volume[i]
            vol_ma[i] = vol_sum / vol_ma_period
    
    # Get 1d data for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    camarilla_R1 = np.full(len(close_1d), np.nan)
    camarilla_S1 = np.full(len(close_1d), np.nan)
    camarilla_H4 = np.full(len(close_1d), np.nan)
    camarilla_L4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if i == 0:
            # For first bar, use same values (no prior day)
            high_val = high_1d[i]
            low_val = low_1d[i]
            close_val = close_1d[i]
        else:
            high_val = high_1d[i-1]  # Previous day's high
            low_val = low_1d[i-1]    # Previous day's low
            close_val = close_1d[i-1] # Previous day's close
        
        if not (np.isnan(high_val) or np.isnan(low_val) or np.isnan(close_val)):
            range_val = high_val - low_val
            camarilla_R1[i] = close_val + (range_val * 1.1 / 12)
            camarilla_S1[i] = close_val - (range_val * 1.1 / 12)
            camarilla_H4[i] = close_val + (range_val * 1.1 / 2)
            camarilla_L4[i] = close_val - (range_val * 1.1 / 2)
    
    # Align Camarilla levels to 12h timeframe
    camarilla_R1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_R1)
    camarilla_S1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_S1)
    camarilla_H4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_H4)
    camarilla_L4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_L4)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 for trend filter
    ema_1w_period = 50
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_1w_period:
        ema_1w[ema_1w_period - 1] = np.mean(close_1w[:ema_1w_period])
        for i in range(ema_1w_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_1w_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_1w_period + 1))))
    
    # Align 1w EMA50 to 12h timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need volume MA, Camarilla levels, and EMA1w
    start_idx = max(vol_ma_period - 1, 0)  # Camarilla starts from index 0 aligned
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vol_ma[i]) or np.isnan(camarilla_R1_aligned[i]) or 
            np.isnan(camarilla_S1_aligned[i]) or np.isnan(camarilla_H4_aligned[i]) or
            np.isnan(camarilla_L4_aligned[i]) or np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma_val = vol_ma[i]
        r1 = camarilla_R1_aligned[i]
        s1 = camarilla_S1_aligned[i]
        h4 = camarilla_H4_aligned[i]
        l4 = camarilla_L4_aligned[i]
        ema1w_val = ema_1w_aligned[i]
        
        if position == 0:
            # Long: price breaks above R1 with volume confirmation and 1w uptrend
            if (price > r1 and vol > 1.5 * vol_ma_val and price > ema1w_val):
                signals[i] = size
                position = 1
            # Short: price breaks below S1 with volume confirmation and 1w downtrend
            elif (price < s1 and vol > 1.5 * vol_ma_val and price < ema1w_val):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to H4 level or trend fails
            if price >= h4 or price < ema1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to L4 level or trend fails
            if price <= l4 or price > ema1w_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_1wTrend_Volume"
timeframe = "12h"
leverage = 1.0