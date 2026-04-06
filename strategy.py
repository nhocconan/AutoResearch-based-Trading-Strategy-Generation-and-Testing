#!/usr/bin/env python3
"""
6h Camarilla Pivot Reversal with Volume Confirmation
Hypothesis: Camarilla pivot levels (R3, S3) act as strong reversal zones in mean-reverting markets,
while breaks of R4/S4 indicate continuation. Using 1d pivot levels on 6h timeframe filters noise
and captures reversals in both bull and bear markets. Volume confirmation ensures institutional
participation. Target: 75-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_camarilla_pivot_rev_v1"
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
    
    # 14-period ATR for stops and filters
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
    
    # Calculate Camarilla pivot levels for each 1d bar
    # R4 = close + ((high - low) * 1.1/2)
    # R3 = close + ((high - low) * 1.1/4)
    # S3 = close - ((high - low) * 1.1/4)
    # S4 = close - ((high - low) * 1.1/2)
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if not (np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i])):
            diff = high_1d[i] - low_1d[i]
            camarilla_r4[i] = close_1d[i] + (diff * 1.1 / 2)
            camarilla_r3[i] = close_1d[i] + (diff * 1.1 / 4)
            camarilla_s3[i] = close_1d[i] - (diff * 1.1 / 4)
            camarilla_s4[i] = close_1d[i] - (diff * 1.1 / 2)
    
    # Align pivot levels to 6h timeframe (shifted by 1 for completed bars only)
    r4_6h = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_6h = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_6h = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_6h = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume filter: current volume > 1.5x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(r3_6h[i]) or np.isnan(s3_6h[i]) or \
           np.isnan(r4_6h[i]) or np.isnan(s4_6h[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches S3 (mean reversion target) OR stoploss hit
            # Stoploss: price drops 2.0*ATR below entry
            if (close[i] <= s3_6h[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R3 (mean reversion target) OR stoploss hit
            # Stoploss: price rises 2.0*ATR above entry
            if (close[i] >= r3_6h[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for reversal entries at extreme levels
            # Long: price rejects S4 and moves back above S3 with volume
            if (close[i] > s3_6h[i] and
                close[i-1] <= s4_6h[i] and  # was at or below S4
                volume_filter):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: price rejects R4 and moves back below R3 with volume
            elif (close[i] < r3_6h[i] and
                  close[i-1] >= r4_6h[i] and  # was at or above R4
                  volume_filter):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals