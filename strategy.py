#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous day
    camarilla_h5 = np.full(n, np.nan)  # R4
    camarilla_h4 = np.full(n, np.nan)  # R3
    camarilla_h3 = np.full(n, np.nan)  # R2
    camarilla_h2 = np.full(n, np.nan)  # R1
    camarilla_h1 = np.full(n, np.nan)  # PP
    camarilla_l1 = np.full(n, np.nan)  # S1
    camarilla_l2 = np.full(n, np.nan)  # S2
    camarilla_l3 = np.full(n, np.nan)  # S3
    camarilla_l4 = np.full(n, np.nan)  # S4
    
    # Calculate pivots for each day (starting from index 1)
    for i in range(1, len(high_1d)):
        # Previous day's OHLC
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        range_val = ph - pl
        
        if range_val > 0:
            camarilla_h5[i] = pc + range_val * 1.5  # R4
            camarilla_h4[i] = pc + range_val * 1.25  # R3
            camarilla_h3[i] = pc + range_val * 1.1   # R2
            camarilla_h2[i] = pc + range_val * 0.55  # R1
            camarilla_h1[i] = (ph + pl + pc) / 3     # PP
            camarilla_l1[i] = camarilla_h1[i] - range_val * 0.55  # S1
            camarilla_l2[i] = camarilla_h1[i] - range_val * 1.1   # S2
            camarilla_l3[i] = camarilla_h1[i] - range_val * 1.25  # S3
            camarilla_l4[i] = camarilla_h1[i] - range_val * 1.5   # S4
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 day to avoid look-ahead)
    camarilla_h5_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_h2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h2)
    camarilla_h1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h1)
    camarilla_l1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l1)
    camarilla_l2_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l2)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume filter: current volume > 1.5x average over last 24 periods (4 days)
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 24)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(camarilla_h5_aligned[i]) or 
            np.isnan(camarilla_l4_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below S3 (camarilla_l3) or stoploss hit
            if (close[i] < camarilla_l3_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above R3 (camarilla_h3) or stoploss hit
            if (close[i] > camarilla_h3_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price breaks above R3 (camarilla_h4) with volume
            if (close[i] > camarilla_h4_aligned[i] and volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price breaks below S3 (camarilla_l3) with volume
            elif (close[i] < camarilla_l3_aligned[i] and volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals