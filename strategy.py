#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for ATR and volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h ATR (14-period)
    tr_12h = np.maximum(high_12h - low_12h, 
                        np.maximum(np.abs(high_12h - np.roll(close_12h, 1)), 
                                   np.abs(low_12h - np.roll(close_12h, 1))))
    tr_12h[0] = high_12h[0] - low_12h[0]
    atr14_12h = pd.Series(tr_12h).rolling(window=14, min_periods=14).mean().values
    
    # Calculate 12h volume average (24-period)
    volume_ma24 = pd.Series(volume_12h).rolling(window=24, min_periods=24).mean().values
    
    # Get 1d data for daily pivot points
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily pivot points (standard formula)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    r1_1d = 2 * pivot_1d - low_1d
    s1_1d = 2 * pivot_1d - high_1d
    
    # Align daily pivot levels to 12h timeframe (use previous day's levels)
    pivot_12h = align_htf_to_ltf(prices, df_1d, pivot_1d)
    r1_12h = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_12h = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # Calculate 4h trend filter (EMA34)
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 24  # Need sufficient data for volume MA and EMA
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(pivot_12h[i]) or np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or
            np.isnan(atr14_12h[i]) or np.isnan(volume_ma24[i]) or np.isnan(ema34_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current 12h volume > 1.5 * 24-period average
        volume_filter = volume_12h[i // 2] > (1.5 * volume_ma24[i // 2]) if i >= 2 else False
        
        # Trend filter: price above/below EMA34 on 4h
        trend_filter = close_12h[i // 2] > ema34_12h[i // 2] if i >= 2 else False
        
        if position == 0:
            # Long: price breaks above R1 with volume and uptrend
            if (close[i] > r1_12h[i] and volume_filter and trend_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with volume and downtrend
            elif (close[i] < s1_12h[i] and volume_filter and not trend_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls below S1 or trend turns down
            if close[i] < s1_12h[i] or not trend_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises above R1 or trend turns up
            if close[i] > r1_12h[i] or trend_filter:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Pivot_Breakout_Volume_TrendFilter"
timeframe = "12h"
leverage = 1.0