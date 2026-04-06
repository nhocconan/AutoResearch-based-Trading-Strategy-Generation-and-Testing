#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1D Tenkan-Kijun Cross + Volume Spike
Hypothesis: Uses 1D Ichimoku (Tenkan/Kijun cross, price vs cloud) for trend bias,
combined with 6D price action and volume confirmation to enter trades.
Works in bull (price above cloud, bullish cross, volume) and bear (price below cloud, bearish cross, volume).
Designed for low trade frequency (~15-30/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_ata, align_htf_to_ltf

name = "6h_ichimoku_1dtkx_vol"
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
    
    # 2-period ATR for stoploss
    atr = np.full(n, np.nan)
    if n >= 2:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[1] = tr[0]
            for i in range(2, n):
                atr[i] = (tr[i-1] + atr[i-1]) / 2  # Wilder's smoothing
    
    # 1D Ichimoku components
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Tenkan-sen (Conversion Line): (9-period high + low)/2
    tenkan = np.full(len(high_1d), np.nan)
    if len(high_1d) >= 9:
        for i in range(8, len(high_1d)):
            tenkan[i] = (np.max(high_1d[i-8:i+1]) + np.min(low_1d[i-8:i+1])) / 2
    
    # Kijun-sen (Base Line): (26-period high + low)/2
    kijun = np.full(len(high_1d), np.nan)
    if len(high_1d) >= 26:
        for i in range(25, len(high_1d)):
            kijun[i] = (np.max(high_1d[i-25:i+1]) + np.min(low_1d[i-25:i+1])) / 2
    
    # Senkou Span A (Leading Span A): (Tenkan + Kijun)/2 shifted 26 periods ahead
    senkou_a = np.full(len(high_1d), np.nan)
    if len(high_1d) >= 52:  # Need 26+26 for calculation
        for i in range(26, len(high_1d)):
            if not np.isnan(tenkan[i-26]) and not np.isnan(kijun[i-26]):
                senkou_a[i] = (tenkan[i-26] + kijun[i-26]) / 2
    
    # Senkou Span B (Leading Span B): (52-period high + low)/2 shifted 26 periods ahead
    senkou_b = np.full(len(high_1d), np.nan)
    if len(high_1d) >= 78:  # Need 52+26 for calculation
        for i in range(52, len(high_1d)):
            if i-26 >= 0:
                high_52 = np.max(high_1d[i-52:i-26+1]) if i-52 >= 0 else np.max(high_1d[:i-26+1])
                low_52 = np.min(low_1d[i-52:i-26+1]) if i-52 >= 0 else np.min(low_1d[:i-26+1])
                senkou_b[i] = (high_52 + low_52) / 2
    
    # Align Ichimoku components to 6h timeframe
    tenkan_aligned = align_htf_to_ltf(prices, df_1d, tenkan)
    kijun_aligned = align_htf_to_ltf(prices, df_1d, kijun)
    senkou_a_aligned = align_htf_to_ltf(prices, df_1d, senkou_a)
    senkou_b_aligned = align_htf_to_ltf(prices, df_1d, senkou_b)
    
    # Cloud top and bottom
    cloud_top = np.maximum(senkou_a_aligned, senkou_b_aligned)
    cloud_bottom = np.minimum(senkou_a_aligned, senkou_b_aligned)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period (need 52 periods for Senkou B)
    start = 52
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(atr[i]) or np.isnan(tenkan_aligned[i]) or np.isnan(kijun_aligned[i]) or \
           np.isnan(cloud_top[i]) or np.isnan(cloud_bottom[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter (20-period average)
        vol_ma = np.mean(volume[max(0, i-20):i+1])
        volume_filter = volume[i] > vol_ma * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below cloud OR Tenkan-Kijun cross turns bearish
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < cloud_bottom[i] or
                tenkan_aligned[i] < kijun_aligned[i] or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price closes above cloud OR Tenkan-Kijun cross turns bullish
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > cloud_top[i] or
                tenkan_aligned[i] > kijun_aligned[i] or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries: Ichimoku signals + volume
            # Minimum holding period: only allow new entry after 30 bars flat
            if bars_since_entry >= 30:
                # Bullish: price above cloud, bullish TK cross, volume
                if (close[i] > cloud_top[i] and
                    tenkan_aligned[i] > kijun_aligned[i] and
                    volume_filter):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Bearish: price below cloud, bearish TK cross, volume
                elif (close[i] < cloud_bottom[i] and
                      tenkan_aligned[i] < kijun_aligned[i] and
                      volume_filter):
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