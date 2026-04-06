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
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla pivot levels for previous day
    camarilla_h4 = np.full(n, np.nan)  # H4 level
    camarilla_l4 = np.full(n, np.nan)  # L4 level
    camarilla_h3 = np.full(n, np.nan)  # H3 level
    camarilla_l3 = np.full(n, np.nan)  # L3 level
    
    # Calculate pivots for each 12h bar based on previous day
    for i in range(n):
        if i == 0:
            continue
        # Get previous day's OHLC
        prev_day_idx = i - 1
        # Simplified: use current day's data as proxy for previous day in 12h context
        # For proper alignment, we use the 1d data shifted by 1
        if prev_day_idx < len(high_1d):
            ph = high_1d[prev_day_idx]
            pl = low_1d[prev_day_idx]
            pc = close_1d[prev_day_idx]
            if not (np.isnan(ph) or np.isnan(pl) or np.isnan(pc)):
                rang = ph - pl
                camarilla_h4[i] = pc + 1.1 * rang / 2
                camarilla_l4[i] = pc - 1.1 * rang / 2
                camarilla_h3[i] = pc + 1.1 * rang / 4
                camarilla_l3[i] = pc - 1.1 * rang / 4
    
    # Align Camarilla levels to 12h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    # Volume filter: current volume > 1.5x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(30, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(camarilla_h4_aligned[i]) or 
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
            # Exit: price closes below L3 or stoploss hit
            if (close[i] < camarilla_l3_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price closes above H3 or stoploss hit
            if (close[i] > camarilla_h3_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long: price crosses above H4 with volume
            if (close[i] > camarilla_h4_aligned[i] and 
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price crosses below L4 with volume
            elif (close[i] < camarilla_l4_aligned[i] and 
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals