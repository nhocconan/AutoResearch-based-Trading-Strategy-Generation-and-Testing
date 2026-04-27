#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian(20) breakout with 1-week ADX trend filter and volume confirmation.
Trades breakouts when price breaks above/below 20-day Donchian channel, ADX > 25 confirms trend strength,
and volume > 1.5x 20-day average volume confirms breakout strength.
Designed for trending markets with clear breakouts, works in both bull and bear by using ADX to filter.
Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1-day data for Donchian calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1-day Donchian(20) channels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Upper band: 20-day high
    upper_1d = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-day low
    lower_1d = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1-day timeframe
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_1d)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_1d)
    
    # Get 1-week data for ADX calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX(14) on 1-week data
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    
    # Directional Movement
    up_move = np.diff(high_1w, prepend=high_1w[0])
    down_move = -np.diff(low_1w, prepend=low_1w[0])
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    def smma(arr, period):
        result = np.zeros_like(arr)
        result[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            result[i] = (result[i-1] * (period-1) + arr[i]) / period
        return result
    
    atr = smma(tr, 14)
    plus_di = 100 * smma(plus_dm, 14) / atr
    minus_di = 100 * smma(minus_dm, 14) / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = smma(dx, 14)
    
    # Align ADX to 1-day timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Get 1-day data for volume filter
    vol_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian levels, ADX, and volume MA
    start_idx = max(20, 30, 20)  # max of lookbacks
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_1d_aligned[i]
        adx_now = adx_aligned[i]
        
        # Current Donchian levels
        upper_now = upper_aligned[i]
        lower_now = lower_aligned[i]
        
        # Volume filter: volume > 1.5x 20-day average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # ADX filter: ADX > 25 indicates strong trend
        trend_filter = adx_now > 25
        
        # Entry conditions: Donchian breakout with volume and trend confirmation
        if position == 0:
            # Long: price breaks above upper Donchian with volume + trend
            if price_now > upper_now and vol_filter and trend_filter:
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian with volume + trend
            elif price_now < lower_now and vol_filter and trend_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to lower Donchian or ADX weakens
            if price_now < lower_now or adx_now < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to upper Donchian or ADX weakens
            if price_now > upper_now or adx_now < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_Donchian20_Breakout_1wADX_Volume"
timeframe = "1d"
leverage = 1.0