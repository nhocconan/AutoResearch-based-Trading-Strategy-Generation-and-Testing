#!/usr/bin/env python3
"""
6h_WeeklyPivot_R1_S1_Breakout_1dTrend_VolumeFilter_v1
Hypothesis: 6h breakouts at weekly Camarilla R1/S1 levels, filtered by 1d EMA50 trend and volume spike (>2x 24-period average).
Weekly pivots capture major support/resistance from prior week, effective in both bull (breakout continuation) and bear (mean reversion at extremes) markets.
Discrete position sizing (0.25) and wide ATR stop (3.0x) reduce fee drag and whipsaw. Target: 12-30 trades/year per symbol.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA50, 1w for weekly pivots)
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 60 or len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    df_1d_close = df_1d['close'].values
    ema_50_1d = pd.Series(df_1d_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === Weekly OHLC for Camarilla pivot calculation (based on previous 1w bar) ===
    df_1w_open = df_1w['open'].values
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    # Calculate Camarilla levels for each 1w bar
    range_1w = df_1w_high - df_1w_low
    r1_1w = df_1w_close + 0.275 * range_1w
    s1_1w = df_1w_close - 0.275 * range_1w
    
    # Align 1w Camarilla levels to 6h timeframe
    r1_1w_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_1w_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    # === ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume filter: 24-period average (4 days on 6h) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r1_1w_aligned[i]) or np.isnan(s1_1w_aligned[i]) 
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol_current = volume[i]
        vol_average = vol_ma[i]
        
        if position == 0:
            # Volume filter: current volume > 2.0x 24-period average
            vol_filter = vol_current > 2.0 * vol_average
            
            # Long conditions: price > weekly R1 (breakout), 1d uptrend, volume filter
            long_breakout = price > r1_1w_aligned[i]
            long_trend = price > ema_50_1d_aligned[i]
            
            # Short conditions: price < weekly S1 (breakdown), 1d downtrend, volume filter
            short_breakout = price < s1_1w_aligned[i]
            short_trend = price < ema_50_1d_aligned[i]
            
            # Entry logic - filters for quality trades
            if long_breakout and long_trend and vol_filter:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_breakout and short_trend and vol_filter:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (wide 3.0x ATR to avoid premature exits in volatile markets)
            if price < entry_price - 3.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes below weekly S1 (breakdown)
            elif price < s1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (wide 3.0x ATR)
            if price > entry_price + 3.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trailing exit: price closes above weekly R1 (breakout)
            elif price > r1_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_WeeklyPivot_R1_S1_Breakout_1dTrend_VolumeFilter_v1"
timeframe = "6h"
leverage = 1.0