#!/usr/bin/env python3
"""
12h_Camarilla_Pivot_Breakout_Volume_Regime
- Long/short at Camarilla pivot R1/S1 levels with volume confirmation
- Trend filter: 1d EMA34 direction
- Regime filter: 1d Choppiness Index > 61.8 (range) for mean-reversion, < 38.2 (trend) for breakout
- Position size: 0.25
- Exit on opposite Camarilla level (R1 for longs, S1 for shorts)
- Designed for 12-30 trades/year per symbol
Works in bull (breakouts) and bear (mean reversion in range) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(arr, period):
    """Calculate Exponential Moving Average."""
    if len(arr) < period:
        return np.full(len(arr), np.nan)
    
    ema = np.full(len(arr), np.nan)
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(arr[:period])
    
    for i in range(period, len(arr)):
        ema[i] = (arr[i] * multiplier) + (ema[i-1] * (1 - multiplier))
    
    return ema

def calculate_choppiness(high, low, close, period):
    """Calculate Choppiness Index."""
    if len(high) < period:
        return np.full(len(high), np.nan)
    
    atr = np.zeros(len(high))
    for i in range(1, len(high)):
        tr1 = high[i] - low[i]
        tr2 = abs(high[i] - close[i-1])
        tr3 = abs(low[i] - close[i-1])
        atr[i] = max(tr1, tr2, tr3)
    
    # Sum of true range over period
    tr_sum = np.zeros(len(high))
    for i in range(period-1, len(high)):
        tr_sum[i] = np.sum(atr[i-period+1:i+1])
    
    # Highest high and lowest low over period
    hh = np.zeros(len(high))
    ll = np.zeros(len(high))
    for i in range(period-1, len(high)):
        hh[i] = np.max(high[i-period+1:i+1])
        ll[i] = np.min(low[i-period+1:i+1])
    
    # Choppiness formula
    chop = np.full(len(high), np.nan)
    for i in range(period-1, len(high)):
        if hh[i] != ll[i]:
            chop[i] = 100 * np.log10(tr_sum[i] / (hh[i] - ll[i])) / np.log10(period)
    
    return chop

def calculate_camarilla(high, low, close):
    """Calculate Camarilla pivot levels."""
    # Typical price
    typical = (high + low + close) / 3
    range_val = high - low
    
    R1 = close + range_val * 1.1 / 12
    S1 = close - range_val * 1.1 / 12
    R2 = close + range_val * 1.1 / 6
    S2 = close - range_val * 1.1 / 6
    R3 = close + range_val * 1.1 / 4
    S3 = close - range_val * 1.1 / 4
    R4 = close + range_val * 1.1 / 2
    S4 = close - range_val * 1.1 / 2
    
    return R1, S1, R2, S2, R3, S3, R4, S4

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA, ATR, and Choppiness
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = calculate_ema(close_1d, 34)
    
    # Calculate 1d Choppiness Index (14-period)
    chop_14_1d = calculate_choppiness(high_1d, low_1d, close_1d, 14)
    
    # Calculate Camarilla levels from 1d OHLC
    R1_1d, S1_1d, R2_1d, S2_1d, R3_1d, S3_1d, R4_1d, S4_1d = calculate_camarilla(
        high_1d, low_1d, close_1d
    )
    
    # Align 1d indicators to 12h timeframe
    ema_34_1d_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    chop_14_1d_12h = align_htf_to_ltf(prices, df_1d, chop_14_1d)
    R1_1d_12h = align_htf_to_ltf(prices, df_1d, R1_1d)
    S1_1d_12h = align_htf_to_ltf(prices, df_1d, S1_1d)
    R2_1d_12h = align_htf_to_ltf(prices, df_1d, R2_1d)
    S2_1d_12h = align_htf_to_ltf(prices, df_1d, S2_1d)
    
    # Volume moving average (20-period)
    vol_ma = np.full(len(volume), np.nan)
    if len(volume) >= 20:
        for i in range(19, len(volume)):
            vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # need sufficient data for EMA, Choppiness, and volume MA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1d_12h[i]) or np.isnan(chop_14_1d_12h[i]) or 
            np.isnan(R1_1d_12h[i]) or np.isnan(S1_1d_12h[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Trend filter: price above/below EMA34
        price_above_ema = close[i] > ema_34_1d_12h[i]
        price_below_ema = close[i] < ema_34_1d_12h[i]
        
        # Regime filter
        chop_value = chop_14_1d_12h[i]
        is_ranging = chop_value > 61.8  # Chop > 61.8 = ranging market
        is_trending = chop_value < 38.2  # Chop < 38.2 = trending market
        
        if position == 0:
            # Long conditions: price at S1 level + volume + (trending OR ranging with mean reversion)
            at_s1 = abs(close[i] - S1_1d_12h[i]) < (S1_1d_12h[i] * 0.001)  # within 0.1%
            
            if at_s1 and vol_filter:
                # In trending market: go with trend (long if price above EMA)
                # In ranging market: mean reversion at support (long)
                if (is_trending and price_above_ema) or is_ranging:
                    signals[i] = 0.25
                    position = 1
            
            # Short conditions: price at R1 level + volume + (trending OR ranging with mean reversion)
            at_r1 = abs(close[i] - R1_1d_12h[i]) < (R1_1d_12h[i] * 0.001)  # within 0.1%
            
            if at_r1 and vol_filter:
                # In trending market: go with trend (short if price below EMA)
                # In ranging market: mean reversion at resistance (short)
                if (is_trending and price_below_ema) or is_ranging:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Long exit: price reaches R1 level or closes below S1
            at_r1_exit = abs(close[i] - R1_1d_12h[i]) < (R1_1d_12h[i] * 0.001)
            below_s1 = close[i] < S1_1d_12h[i]
            
            if at_r1_exit or below_s1:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price reaches S1 level or closes above R1
            at_s1_exit = abs(close[i] - S1_1d_12h[i]) < (S1_1d_12h[i] * 0.001)
            above_r1 = close[i] > R1_1d_12h[i]
            
            if at_s1_exit or above_r1:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_Pivot_Breakout_Volume_Regime"
timeframe = "12h"
leverage = 1.0