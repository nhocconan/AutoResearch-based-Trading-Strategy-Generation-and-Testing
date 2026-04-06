#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_r4s4_breakout_v2"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Get 12h data for Camarilla pivot calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Camarilla levels from previous 12h bar
    # R4 = C + ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4)
    # S3 = C - ((H-L) * 1.1/4)
    # S4 = C - ((H-L) * 1.1/2)
    camarilla_r4 = np.full(len(close_12h), np.nan)
    camarilla_r3 = np.full(len(close_12h), np.nan)
    camarilla_s3 = np.full(len(close_12h), np.nan)
    camarilla_s4 = np.full(len(close_12h), np.nan)
    
    for i in range(1, len(close_12h)):
        prev_high = high_12h[i-1]
        prev_low = low_12h[i-1]
        prev_close = close_12h[i-1]
        range_val = prev_high - prev_low
        
        camarilla_r4[i] = prev_close + (range_val * 1.1 / 2)
        camarilla_r3[i] = prev_close + (range_val * 1.1 / 4)
        camarilla_s3[i] = prev_close - (range_val * 1.1 / 4)
        camarilla_s4[i] = prev_close - (range_val * 1.1 / 2)
    
    # Align Camarilla levels to 6h timeframe
    r4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r4)
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    s4_12h_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s4)
    
    # Volume filter: current volume > 1.5x average over last 24 periods (4 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(100, 24)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(r4_12h_aligned[i]) or 
            np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or 
            np.isnan(s4_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below S3 or stoploss hit
            if (close[i] < s3_12h_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above R3 or stoploss hit
            if (close[i] > r3_12h_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price breaks above R4 with volume (breakout continuation)
            if (close[i] > r4_12h_aligned[i] and volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below S4 with volume (breakout continuation)
            elif (close[i] < s4_12h_aligned[i] and volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals