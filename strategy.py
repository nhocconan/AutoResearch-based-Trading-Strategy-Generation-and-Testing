#!/usr/bin/env python3
"""
6h_WeeklyPivot_Donchian20_Breakout_ATRFilter_v1
Hypothesis: 6h Donchian(20) breakout filtered by weekly pivot direction and ATR-based volatility regime.
In bullish weekly bias (price > weekly pivot): long Donchian breakouts, avoid shorts.
In bearish weekly bias (price < weekly pivot): short Donchian breakouts, avoid longs.
ATR filter ensures breakouts occur during sufficient volatility to avoid false signals.
Uses discrete position sizing (0.25) to manage drawdown and minimize fee churn.
Designed for 6h timeframe with 1w HTF for weekly pivot and 1d for ATR regime.
Target: 50-150 total trades over 4 years = 12-37/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1w for weekly pivot, 1d for ATR regime)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1w OHLC for weekly pivot calculation (based on previous 1w bar) ===
    df_1w_open = df_1w['open'].values
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # Calculate weekly pivot points (standard formula)
    weekly_pivot = (df_1w_high + df_1w_low + df_1w_close) / 3.0
    weekly_r1 = 2 * weekly_pivot - df_1w_low
    weekly_s1 = 2 * weekly_pivot - df_1w_high
    weekly_r2 = weekly_pivot + (df_1w_high - df_1w_low)
    weekly_s2 = weekly_pivot - (df_1w_high - df_1w_low)
    
    # Align 1w weekly pivot to 6h timeframe
    weekly_pivot_aligned = align_htf_to_ltf(prices, df_1w, weekly_pivot)
    weekly_r1_aligned = align_htf_to_ltf(prices, df_1w, weekly_r1)
    weekly_s1_aligned = align_htf_to_ltf(prices, df_1w, weekly_s1)
    weekly_r2_aligned = align_htf_to_ltf(prices, df_1w, weekly_r2)
    weekly_s2_aligned = align_htf_to_ltf(prices, df_1w, weekly_s2)
    
    # === 1d ATR(14) for volatility regime filter ===
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # True Range calculation
    tr1 = pd.Series(df_1d_high - df_1d_low)
    tr2 = pd.Series(np.abs(df_1d_high - np.roll(df_1d_close, 1)))
    tr3 = pd.Series(np.abs(df_1d_low - np.roll(df_1d_close, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Align 1d ATR to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    # === 6h Donchian(20) breakout levels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian channels (20-period)
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === 6h ATR(14) for stoploss and volatility filter ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_6h = tr.rolling(window=14, min_periods=14).mean().values
    
    # === 6h Volume(20) for confirmation ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(weekly_pivot_aligned[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) 
            or np.isnan(atr_6h[i]) or np.isnan(atr_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        weekly_pivot = weekly_pivot_aligned[i]
        weekly_r1 = weekly_r1_aligned[i]
        weekly_s1 = weekly_s1_aligned[i]
        weekly_r2 = weekly_r2_aligned[i]
        weekly_s2 = weekly_s2_aligned[i]
        upper_donchian = donchian_high[i]
        lower_donchian = donchian_low[i]
        atr_now = atr_6h[i]
        atr_regime = atr_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volatility filter: current 6h ATR > 0.5x 1d ATR (ensures sufficient volatility)
        vol_filter = atr_now > 0.5 * atr_regime
        
        # Volume confirmation: current volume > 1.2x average
        volume_confirmed = volume_now > 1.2 * vol_avg
        
        if position == 0:
            # Weekly bias filter: only trade in direction of weekly pivot
            bullish_bias = price > weekly_pivot
            bearish_bias = price < weekly_pivot
            
            # Long conditions: price breaks above Donchian high + bullish bias + filters
            long_condition = (price > upper_donchian) and bullish_bias and vol_filter and volume_confirmed
            
            # Short conditions: price breaks below Donchian low + bearish bias + filters
            short_condition = (price < lower_donchian) and bearish_bias and vol_filter and volume_confirmed
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr_now:
                signals[i] = 0.0
                position = 0
            # Exit if price re-enters Donchian channel (breakout failed)
            elif price < upper_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr_now:
                signals[i] = 0.0
                position = 0
            # Exit if price re-enters Donchian channel (breakout failed)
            elif price > lower_donchian:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_Donchian20_Breakout_ATRFilter_v1"
timeframe = "6h"
leverage = 1.0