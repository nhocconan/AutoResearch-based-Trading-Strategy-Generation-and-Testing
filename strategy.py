#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with daily Camarilla pivot filter + volume confirmation
    # Long: price > Donchian(20) high AND price > daily S3 pivot AND volume > 2.0x 20-period average
    # Short: price < Donchian(20) low AND price < daily R3 pivot AND volume > 2.0x 20-period average
    # Exit: opposite Donchian breakout OR price crosses daily H3/L3 pivot
    # Using 12h timeframe for low trade frequency (target 12-37/year), daily Camarilla pivots for strong structure,
    # and volume spike confirmation to avoid false breakouts. Discrete position sizing (0.25) to minimize fee churn.
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot levels (H3, L3, S3, R3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels with proper seeding
    H4 = np.full(len(close_1d), np.nan)
    L4 = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        H4[i] = close_1d[i-1] + 1.1 * (high_1d[i-1] - low_1d[i-1]) * 1.1 / 2
        L4[i] = close_1d[i-1] - 1.1 * (high_1d[i-1] - low_1d[i-1]) * 1.1 / 2
    
    H3 = np.full(len(close_1d), np.nan)
    L3 = np.full(len(close_1d), np.nan)
    S3 = np.full(len(close_1d), np.nan)
    R3 = np.full(len(close_1d), np.nan)
    for i in range(1, len(close_1d)):
        H3[i] = H4[i] - (H4[i] - L4[i]) / 2
        L3[i] = L4[i] + (H4[i] - L4[i]) / 2
        S3[i] = L4[i] - (H4[i] - L4[i]) * 1.1 / 6
        R3[i] = H4[i] + (H4[i] - L4[i]) * 1.1 / 6
    
    # Get 12h Donchian(20) for breakout with min_periods
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Get 12h volume for confirmation (>2.0x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (2.0 * vol_ma)
    
    # Align daily Camarilla levels to 12h
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    S3_aligned = align_htf_to_ltf(prices, df_1d, S3)
    R3_aligned = align_htf_to_ltf(prices, df_1d, R3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(S3_aligned[i]) or np.isnan(R3_aligned[i]) or
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # Pivot filter conditions
        bullish_pivot = close[i] > S3_aligned[i]  # Above S3 = bullish bias
        bearish_pivot = close[i] < R3_aligned[i]  # Below R3 = bearish bias
        
        # Entry logic: Breakout + pivot alignment + volume confirmation
        long_entry = long_breakout and bullish_pivot and volume_spike[i]
        short_entry = short_breakout and bearish_pivot and volume_spike[i]
        
        # Exit logic: opposite breakout or price crosses H3/L3 (more significant levels)
        long_exit = short_breakout or (close[i] < L3_aligned[i])
        short_exit = long_breakout or (close[i] > H3_aligned[i])
        
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

name = "12h_1d_donchian_breakout_camarilla_pivot_volume_v1"
timeframe = "12h"
leverage = 1.0