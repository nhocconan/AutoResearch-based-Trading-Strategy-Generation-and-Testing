#!/usr/bin/env python3
"""
6h Donchian(20) Breakout + Weekly Pivot + Volume Spike
Hypothesis: Donchian breakouts with volume spike (>2x average) in the direction of weekly pivot (above weekly pivot = long, below = short) capture high-probability trend moves. Weekly pivot provides structural bias from higher timeframe, reducing whipsaws. Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_donchian20_weekly_pivot_vol_v2"
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
    
    # 14-period ATR
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] * 13 + atr[i-1]) / 14
    
    # Weekly pivot levels (using 1w data as HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) >= 3:
        # Calculate weekly pivot: P = (H + L + C)/3
        # R1 = 2*P - L, S1 = 2*P - H
        # R2 = P + (H - L), S2 = P - (H - L)
        # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
        high_w = df_1w['high'].values
        low_w = df_1w['low'].values
        close_w = df_1w['close'].values
        
        pivot_w = (high_w + low_w + close_w) / 3.0
        r1_w = 2 * pivot_w - low_w
        s1_w = 2 * pivot_w - high_w
        r2_w = pivot_w + (high_w - low_w)
        s2_w = pivot_w - (high_w - low_w)
        r3_w = high_w + 2 * (pivot_w - low_w)
        s3_w = low_w - 2 * (high_w - pivot_w)
        
        # For entry bias: use weekly pivot as reference
        # Above pivot = bullish bias, below = bearish bias
        pivot_bias = pivot_w  # Use pivot line itself for simplicity
        
        pivot_bias_aligned = align_htf_to_ltf(prices, df_1w, pivot_bias)
    else:
        pivot_bias_aligned = np.full(n, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 30  # For Donchian(20) + some buffer
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(pivot_bias_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i])
        lowest_low = np.min(low[i-20:i])
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[i-20:i])
        volume_filter = volume[i] > vol_ma * 2.0
        
        # Weekly pivot bias
        pivot_bias_val = pivot_bias_aligned[i]
        above_pivot = close[i] > pivot_bias_val
        below_pivot = close[i] < pivot_bias_val
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below Donchian lower
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < lowest_low or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above Donchian upper
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > highest_high or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Donchian breakout + volume + weekly pivot bias
            # Minimum holding period: only allow new entry after 25 bars flat
            if bars_since_entry >= 25:
                bull_breakout = close[i] > highest_high
                bear_breakout = close[i] < lowest_low
                
                # Long: bullish breakout + volume + above weekly pivot
                if bull_breakout and volume_filter and above_pivot:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: bearish breakout + volume + below weekly pivot
                elif bear_breakout and volume_filter and below_pivot:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
                    bars_since_entry = 0
                else:
                    signals[i] = 0.0
                    bars_since_entry += 1
            else:
                signals[i] = 0.0
                bars_since_entry += 1
    
    return signals