#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 6h strategy using 1d Camarilla pivot levels (R3/S3 for mean reversion, R4/S4 for breakout)
    # Combined with 1w trend filter (price vs 1w EMA50) and volume confirmation
    # Long mean reversion: price < S3 AND price > 1w EMA50 (bullish 1w trend)
    # Short mean reversion: price > R3 AND price < 1w EMA50 (bearish 1w trend)
    # Long breakout: price > R4 AND price > 1w EMA50
    # Short breakout: price < S4 AND price < 1w EMA50
    # Volume filter: volume > 1.5x 20-period average to avoid low-volatility false signals
    # Discrete position sizing: 0.25 for entries, 0.0 for flat
    # Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots for 1d (using previous day's OHLC)
    # Camarilla levels: based on previous day's range
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Initialize Camarilla arrays (same length as 1d data)
    R3 = np.full(len(close_1d), np.nan)
    S3 = np.full(len(close_1d), np.nan)
    R4 = np.full(len(close_1d), np.nan)
    S4 = np.full(len(close_1d), np.nan)
    PP = np.full(len(close_1d), np.nan)  # Pivot point
    
    for i in range(1, len(close_1d)):
        # Previous day's OHLC
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        
        # Pivot point
        PP[i] = (prev_high + prev_low + prev_close) / 3.0
        
        # Range
        range_val = prev_high - prev_low
        
        # Camarilla levels
        R3[i] = PP[i] + range_val * 1.1 / 4.0
        S3[i] = PP[i] - range_val * 1.1 / 4.0
        R4[i] = PP[i] + range_val * 1.1 / 2.0
        S4[i] = PP[i] - range_val * 1.1 / 2.0
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA50 with min_periods
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 50:
        ema_1w[49] = np.mean(close_1w[:50])  # SMA50 as seed
        multiplier = 2 / (50 + 1)
        for i in range(50, len(close_1w)):
            ema_1w[i] = (close_1w[i] * multiplier) + (ema_1w[i-1] * (1 - multiplier))
    
    # Volume confirmation: >1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma)
    
    # Align HTF indicators to 6h timeframe
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d, S4)
    PP_aligned = align_htf_to_ltf(prices, df_1d, PP)
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(R4_aligned[i]) or np.isnan(S4_aligned[i]) or
            np.isnan(ema_1w_aligned[i]) or np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Mean reversion conditions (fade at R3/S3)
        long_mean_rev = (close[i] < S3_aligned[i]) and (close[i] > ema_1w_aligned[i])
        short_mean_rev = (close[i] > R3_aligned[i]) and (close[i] < ema_1w_aligned[i])
        
        # Breakout conditions (continuation at R4/S4)
        long_breakout = (close[i] > R4_aligned[i]) and (close[i] > ema_1w_aligned[i])
        short_breakout = (close[i] < S4_aligned[i]) and (close[i] < ema_1w_aligned[i])
        
        # Combined entry logic
        long_entry = (long_mean_rev or long_breakout) and volume_spike[i]
        short_entry = (short_mean_rev or short_breakout) and volume_spike[i]
        
        # Exit logic: opposite Camarilla level or 1w EMA cross
        long_exit = (close[i] > R3_aligned[i]) or (close[i] < ema_1w_aligned[i])
        short_exit = (close[i] < S3_aligned[i]) or (close[i] > ema_1w_aligned[i])
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1d_1w_camarilla_pivot_ema50_volume_v1"
timeframe = "6h"
leverage = 1.0