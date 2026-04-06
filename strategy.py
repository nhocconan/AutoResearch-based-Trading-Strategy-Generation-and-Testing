#!/usr/bin/env python3
"""
6h Ichimoku Cloud with 1d trend filter and volume confirmation
Hypothesis: Ichimoku cloud acts as dynamic support/resistance. Use 1d trend (price vs EMA200) for bias, cloud for entry/exit, volume for confirmation. Works in bull (price above cloud + bullish 1d) and bear (price below cloud + bearish 1d). Target: 50-150 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ichimoku_1d_vol_v1"
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
    
    # 14-period ATR for stoploss
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
    
    # Get 1d data for trend filter (EMA200)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA200 on 1d close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 198) / 200
    
    # 1d trend: above EMA200 = bullish, below = bearish
    trend_1d = np.where(close_1d > ema_1d, 1, -1)
    
    # Align 1d trend to 6h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d)
    
    # Ichimoku components (6h timeframe)
    # Conversion line (Tenkan-sen): (9-period high + low) / 2
    conv_line = np.full(n, np.nan)
    # Base line (Kijun-sen): (26-period high + low) / 2
    base_line = np.full(n, np.nan)
    # Leading Span A: (Conversion + Base) / 2
    span_a = np.full(n, np.nan)
    # Leading Span B: (52-period high + low) / 2
    span_b = np.full(n, np.nan)
    
    # Calculate Conversion line (9-period)
    for i in range(9, n):
        conv_line[i] = (np.max(high[i-9:i]) + np.min(low[i-9:i])) / 2
    
    # Calculate Base line (26-period)
    for i in range(26, n):
        base_line[i] = (np.max(high[i-26:i]) + np.min(low[i-26:i])) / 2
    
    # Calculate Span B (52-period)
    for i in range(52, n):
        span_b[i] = (np.max(high[i-52:i]) + np.min(low[i-52:i])) / 2
    
    # Calculate Span A (requires Conversion and Base)
    for i in range(26, n):
        if not np.isnan(conv_line[i]) and not np.isnan(base_line[i]):
            span_a[i] = (conv_line[i] + base_line[i]) / 2
    
    # Align Ichimoku components (they are already 6h, but align for consistency)
    # Actually, these are calculated on 6h data, so no alignment needed
    # But we'll keep the variables for clarity
    
    # Get 1d data for volume confirmation
    volume_1d = df_1d['volume'].values
    
    # 20-period average volume on 1d
    vol_ma_1d = np.full(len(volume_1d), np.nan)
    for i in range(20, len(volume_1d)):
        vol_ma_1d[i] = np.mean(volume_1d[i-20:i])
    
    # Align volume MA to 6h timeframe
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start from warmup period
    start = 52  # Need enough data for Span B
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(trend_1d_aligned[i]) or 
            np.isnan(conv_line[i]) or np.isnan(base_line[i]) or
            np.isnan(span_a[i]) or np.isnan(span_b[i]) or
            np.isnan(vol_ma_1d_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            bars_since_entry += 1
            continue
        
        # Volume filter: current 6h volume > 1.5x 1d average volume (scaled)
        # Scale 1d volume to 6h: approx 1/4 of 1d volume (since 4x 6h in 1d)
        vol_threshold = vol_ma_1d_aligned[i] / 4.0 * 1.5
        volume_filter = volume[i] > vol_threshold
        
        # Determine cloud boundaries (Leading Span A and B)
        # Cloud top = max(Span A, Span B), Cloud bottom = min(Span A, Span B)
        cloud_top = max(span_a[i], span_b[i])
        cloud_bottom = min(span_a[i], span_b[i])
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price falls below cloud bottom OR against 1d trend
            # Stoploss: price drops 2*ATR below entry
            if (close[i] < cloud_bottom or
                trend_1d_aligned[i] == -1 or
                close[i] < entry_price - 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25
            bars_since_entry += 1
        elif position == -1:  # short position
            # Exit: price rises above cloud top OR against 1d trend
            # Stoploss: price rises 2*ATR above entry
            if (close[i] > cloud_top or
                trend_1d_aligned[i] == 1 or
                close[i] > entry_price + 2.0 * atr[i]):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = -0.25
            bars_since_entry += 1
        else:
            # Look for entries
            # Minimum holding period: only allow new entry after 4 bars flat
            if bars_since_entry >= 4:
                # Ichimoku signals with 1d trend filter
                price_above_cloud = close[i] > cloud_top
                price_below_cloud = close[i] < cloud_bottom
                
                # Long: price above cloud with bullish 1d trend + volume
                if price_above_cloud and trend_1d_aligned[i] == 1 and volume_filter:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                    bars_since_entry = 0
                # Short: price below cloud with bearish 1d trend + volume
                elif price_below_cloud and trend_1d_aligned[i] == -1 and volume_filter:
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