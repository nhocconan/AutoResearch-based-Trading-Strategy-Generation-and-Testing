#/usr/bin/env python3
"""
12h Camarilla Pivot Reversal with Volume Spike and Chop Filter
Hypothesis: Use daily Camarilla pivot levels (calculated from prior day) on 12h chart with volume confirmation and Choppiness index regime filter to trade mean reversions at strong support/resistance levels. Works in both bull and bear markets by fading extremes at proven institutional levels.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_camarilla_pivot_rev_v1"
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
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from prior day
    camarilla_h5 = np.full(len(close_1d), np.nan)
    camarilla_h4 = np.full(len(close_1d), np.nan)
    camarilla_h3 = np.full(len(close_1d), np.nan)
    camarilla_l3 = np.full(len(close_1d), np.nan)
    camarilla_l4 = np.full(len(close_1d), np.nan)
    camarilla_l5 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Use previous day's data
        ph = high_1d[i-1]
        pl = low_1d[i-1]
        pc = close_1d[i-1]
        range_val = ph - pl
        
        camarilla_h5[i] = pc + range_val * 1.1 / 2
        camarilla_h4[i] = pc + range_val * 1.1 / 4
        camarilla_h3[i] = pc + range_val * 1.1 / 6
        camarilla_l3[i] = pc - range_val * 1.1 / 6
        camarilla_l4[i] = pc - range_val * 1.1 / 4
        camarilla_l5[i] = pc - range_val * 1.1 / 2
    
    # Align Camarilla levels to 12h
    h5_12h = align_htf_to_ltf(prices, df_1d, camarilla_h5)
    h4_12h = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    h3_12h = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_12h = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    l4_12h = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    l5_12h = align_htf_to_ltf(prices, df_1d, camarilla_l5)
    
    # Volume filter: current volume > 2.0x average over last 24 periods
    vol_ma = np.full(n, np.nan)
    for i in range(24, n):
        vol_ma[i] = np.mean(volume[i-24:i])
    
    # Choppiness index (14-period) for regime filter
    chop = np.full(n, np.nan)
    if n >= 14:
        atr_sum = np.full(n, np.nan)
        atr_sum[14] = np.sum(tr[:14]) if len(tr) >= 14 else np.sum(tr[~np.isnan(tr)][:14])
        for i in range(15, n):
            atr_sum[i] = atr_sum[i-1] + tr[i-1]
        
        for i in range(14, n):
            highest = np.max(high[i-14:i+1])
            lowest = np.min(low[i-14:i+1])
            if atr_sum[i] > 0 and (highest - lowest) > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / (highest - lowest)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 24)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(h5_12h[i]) or np.isnan(h4_12h[i]) or np.isnan(h3_12h[i]) or
            np.isnan(l3_12h[i]) or np.isnan(l4_12h[i]) or np.isnan(l5_12h[i]) or np.isnan(vol_ma[i]) or
            np.isnan(chop[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Choppiness regime: Chop > 50 = ranging (good for mean reversion)
        chop_filter = chop[i] > 50
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches L3 or stoploss hit
            if (close[i] <= l3_12h[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches H3 or stoploss hit
            if (close[i] >= h3_12h[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in ranging market with volume
            if chop_filter and volume_filter:
                # Long at L5 with rejection
                if (close[i] <= l5_12h[i] * 1.002 and  # Allow small buffer
                    low[i] <= l5_12h[i] and
                    close[i] > open[i]):  # Bullish close
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short at H5 with rejection
                elif (close[i] >= h5_12h[i] * 0.998 and  # Allow small buffer
                      high[i] >= h5_12h[i] and
                      close[i] < open[i]):  # Bearish close
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals