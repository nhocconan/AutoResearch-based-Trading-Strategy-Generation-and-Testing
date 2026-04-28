#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R1/S1 breakout with 4h trend filter and volume confirmation
# Uses 4h EMA50 for trend direction and 1h for precise entry timing.
# Camarilla R1/S1 breakouts capture intraday momentum with tight stops.
# Session filter (08-20 UTC) reduces noise. Target 60-150 trades over 4 years.
# Designed to work in both bull and bear markets via trend alignment.

name = "1h_Camarilla_R1_S1_Breakout_4hEMA50_Trend_Volume_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA(50) for trend
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align 4h EMA to 1h (changes only when 4h bar closes)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get daily data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivots from previous day OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels: based on previous day's range
    # R1 = close + (high - low) * 1.0/12
    # S1 = close - (high - low) * 1.0/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.0 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.0 / 12
    
    # Align daily Camarilla levels to 1h (use previous day's levels)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour  # prices.index is DatetimeIndex
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # 4h EMA(50) and volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Check session and volume confirmation
        if not session_filter[i] or not volume_confirm[i]:
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        # Handle entries and exits
        if position == 0:  # Flat - look for new entries
            # Long entry: Price > Camarilla R1, above 4h EMA50
            if price > camarilla_r1_aligned[i] and price > ema_50_4h_aligned[i]:
                signals[i] = 0.20
                position = 1
            # Short entry: Price < Camarilla S1, below 4h EMA50
            elif price < camarilla_s1_aligned[i] and price < ema_50_4h_aligned[i]:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:  # Long - exit on retracement to 4h EMA50 or below Camarilla S1
            if price < ema_50_4h_aligned[i] or price < camarilla_s1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # Short - exit on retracement to 4h EMA50 or above Camarilla R1
            if price > ema_50_4h_aligned[i] or price > camarilla_r1_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals