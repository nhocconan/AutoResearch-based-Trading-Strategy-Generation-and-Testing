#!/usr/bin/env python3
"""
1h_Camarilla_Pivot_Trend_Volume
Hypothesis: Camarilla pivot levels (H3/L3, H4/L4) provide strong support/resistance.
In trending markets (identified by 4h EMA20 > EMA50 for long, EMA20 < EMA50 for short),
price tends to continue after retracing to H3/L3 with volume confirmation.
In ranging markets, price reverses at H4/L4. Uses 1d trend filter to avoid counter-trend trades.
Designed for low trade frequency (15-30/year) by requiring confluence of trend, level, and volume.
Works in both bull and bear markets by adapting to trend direction.
"""

name = "1h_Camarilla_Pivot_Trend_Volume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels"""
    range_val = high - low
    pivot = (high + low + close) / 3.0
    h3 = pivot + (range_val * 1.1 / 4)
    l3 = pivot - (range_val * 1.1 / 4)
    h4 = pivot + (range_val * 1.1 / 2)
    l4 = pivot - (range_val * 1.1 / 2)
    return h3, l3, h4, l4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Calculate 4h EMA20 and EMA50 for trend direction
    close_4h = df_4h['close'].values
    ema20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h EMAs to 1h timeframe
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # Calculate Camarilla levels from 4h data
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h_for_cam = df_4h['close'].values
    h3, l3, h4, l4 = calculate_camarilla(high_4h, low_4h, close_4h_for_cam)
    
    # Align Camarilla levels to 1h timeframe
    h3_1h = align_htf_to_ltf(prices, df_4h, h3)
    l3_1h = align_htf_to_ltf(prices, df_4h, l3)
    h4_1h = align_htf_to_ltf(prices, df_4h, h4)
    l4_1h = align_htf_to_ltf(prices, df_4h, l4)
    
    # Get 1d data for trend filter (avoid counter-trend trades)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for higher timeframe trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: > 1.3x 24-period average (to reduce noise)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (1.3 * vol_ma)
    
    # Session filter: 08-20 UTC to avoid low-volume Asian session
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if outside trading session
        if not session_filter[i]:
            signals[i] = 0.0
            continue
            
        # Determine trend direction from 4h EMA crossover
        # Long trend: EMA20 > EMA50, Short trend: EMA20 < EMA50
        long_trend = ema20_4h_aligned[i] > ema50_4h_aligned[i]
        short_trend = ema20_4h_aligned[i] < ema50_4h_aligned[i]
        
        # Higher timeframe trend filter: only trade in direction of 1d trend
        # Long only if price above 1d EMA50, short only if price below 1d EMA50
        long_htf_filter = close[i] > ema50_1d_aligned[i]
        short_htf_filter = close[i] < ema50_1d_aligned[i]
        
        if position == 0:
            # LONG ENTRY: Price retraces to L3 in uptrend with volume confirmation
            # OR price breaks above H4 with volume confirmation (momentum)
            if (long_trend and long_htf_filter and 
                close[i] <= l3_1h[i] and close[i-1] > l3_1h[i-1] and 
                volume_confirm[i]):
                signals[i] = 0.20
                position = 1
            # SHORT ENTRY: Price retraces to H3 in downtrend with volume confirmation
            # OR price breaks below L4 with volume confirmation (momentum)
            elif (short_trend and short_htf_filter and 
                  close[i] >= h3_1h[i] and close[i-1] < h3_1h[i-1] and 
                  volume_confirm[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 
            # 1. Take profit at H3 (reached target level)
            # 2. Stop loss if price breaks below L4 (trend failure)
            # 3. Exit if trend reverses (EMA20 < EMA50)
            if (close[i] >= h3_1h[i] or 
                close[i] < l4_1h[i] or 
                ema20_4h_aligned[i] < ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT:
            # 1. Take profit at L3 (reached target level)
            # 2. Stop loss if price breaks above H4 (trend failure)
            # 3. Exit if trend reverses (EMA20 > EMA50)
            if (close[i] <= l3_1h[i] or 
                close[i] > h4_1h[i] or 
                ema20_4h_aligned[i] > ema50_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals