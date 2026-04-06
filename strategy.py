#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_v1"
timeframe = "12h"
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
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    camarilla_r4 = np.full(n, np.nan)
    camarilla_r3 = np.full(n, np.nan)
    camarilla_s3 = np.full(n, np.nan)
    camarilla_s4 = np.full(n, np.nan)
    
    for i in range(1, len(high_1d)):
        # Calculate pivot levels from previous day's data
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        pivot = (prev_high + prev_low + prev_close) / 3
        range_val = prev_high - prev_low
        
        camarilla_r4[i] = prev_close + range_val * 1.5
        camarilla_r3[i] = prev_close + range_val * 1.25
        camarilla_s3[i] = prev_close - range_val * 1.25
        camarilla_s4[i] = prev_close - range_val * 1.5
    
    # Align Camarilla levels to 12h timeframe (shifted by 1 day for look-ahead avoidance)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Choppiness index (14-period) for regime filter
    chop = np.full(n, np.nan)
    if n >= 14:
        atr_sum = np.full(n, np.nan)
        atr_sum[13] = np.sum(atr[1:15]) if not np.isnan(atr[1:15]).any() else np.nan
        for i in range(15, n):
            if not np.isnan(atr[i]) and not np.isnan(atr_sum[i-1]):
                atr_sum[i] = atr_sum[i-1] + atr[i] - atr[i-13]
        
        for i in range(14, n):
            if not np.isnan(atr_sum[i]) and atr_sum[i] > 0:
                max_high = np.max(high[i-13:i+1])
                min_low = np.min(low[i-13:i+1])
                if max_high > min_low:
                    chop[i] = 100 * np.log10(atr_sum[i] / (max_high - min_low)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Choppiness regime filter: chop > 50 indicates ranging market (good for mean reversion)
        chop_filter = chop[i] > 50
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches S3 or stoploss hit
            if (close[i] <= camarilla_s3_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R3 or stoploss hit
            if (close[i] >= camarilla_r3_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for mean reversion entries at extreme levels
            # Long: price touches S4 with volume and in choppy market
            if (close[i] <= camarilla_s4_aligned[i] and 
                volume_filter and 
                chop_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price touches R4 with volume and in choppy market
            elif (close[i] >= camarilla_r4_aligned[i] and 
                  volume_filter and 
                  chop_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals