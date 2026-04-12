#!/usr/bin/env python3
"""
1h_4h_1d_Supertrend_Pullback
Hypothesis: Use 4h Supertrend for trend direction, 1d ADX for trend strength, and 1h price pullbacks to EMA21 for entries.
Long in uptrend (Supertrend up, ADX>25) when price pulls back to EMA21; short in downtrend (Supertrend down, ADX>25) when price bounces from EMA21.
Designed to capture trend continuation moves with low-frequency entries (target 15-30 trades/year) to minimize fee drag.
Works in bull markets via trend continuation and in bear markets via shorting downtrends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4h_1d_Supertrend_Pullback"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # === 4H DATA: Supertrend for trend direction ===
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 10:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate ATR(10) for Supertrend
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr4h = np.maximum(tr1, np.maximum(tr2, tr3))
    tr4h[0] = tr1[0]  # First period
    atr_4h = pd.Series(tr4h).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Supertrend calculation
    upper_band = (high_4h + low_4h) / 2 + 3 * atr_4h
    lower_band = (high_4h + low_4h) / 2 - 3 * atr_4h
    
    supertrend = np.zeros_like(close_4h)
    direction = np.ones_like(close_4h)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_4h)):
        if close_4h[i] > upper_band[i-1]:
            direction[i] = 1
        elif close_4h[i] < lower_band[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
            if direction[i] == 1 and lower_band[i] < lower_band[i-1]:
                lower_band[i] = lower_band[i-1]
            if direction[i] == -1 and upper_band[i] > upper_band[i-1]:
                upper_band[i] = upper_band[i-1]
        
        if direction[i] == 1:
            supertrend[i] = lower_band[i]
        else:
            supertrend[i] = upper_band[i]
    
    # Align Supertrend and direction to 1h
    supertrend_4h_aligned = align_htf_to_ltf(prices, df_4h, supertrend)
    direction_4h_aligned = align_htf_to_ltf(prices, df_4h, direction)
    
    # === 1D DATA: ADX for trend strength ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.zeros_like(high_1d)
    minus_dm = np.zeros_like(high_1d)
    
    for i in range(1, len(high_1d)):
        up_move = high_1d[i] - high_1d[i-1]
        down_move = low_1d[i-1] - low_1d[i]
        
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
    
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = tr1[0]
    
    # Smoothing
    atr_1d = pd.Series(tr_1d).ewm(span=14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values / atr_1d
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align ADX to 1h
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # === 1H INDICATORS: EMA21 for pullback entries ===
    ema21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # === SESSION FILTER: 08-20 UTC ===
    # Assuming index is DatetimeIndex
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Skip if not ready
        if (np.isnan(supertrend_4h_aligned[i]) or np.isnan(direction_4h_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(ema21[i])):
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            # Outside session: flatten or hold flat
            if position != 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend conditions
        trend_up = direction_4h_aligned[i] == 1 and adx_1d_aligned[i] > 25
        trend_down = direction_4h_aligned[i] == -1 and adx_1d_aligned[i] > 25
        
        # Entry conditions: pullback to EMA21 in trending market
        long_signal = trend_up and close[i] <= ema21[i] * 1.001 and close[i] >= ema21[i] * 0.999
        short_signal = trend_down and close[i] >= ema21[i] * 0.999 and close[i] <= ema21[i] * 1.001
        
        # Exit conditions: trend weakening or reversal
        exit_long = position == 1 and (adx_1d_aligned[i] < 20 or direction_4h_aligned[i] == -1)
        exit_short = position == -1 and (adx_1d_aligned[i] < 20 or direction_4h_aligned[i] == 1)
        
        # Execute trades
        if long_signal and position != 1:
            position = 1
            signals[i] = 0.20
        elif short_signal and position != -1:
            position = -1
            signals[i] = -0.20
        elif exit_long and position == 1:
            position = 0
            signals[i] = 0.0
        elif exit_short and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            signals[i] = 0.20 if position == 1 else (-0.20 if position == -1 else 0.0)
    
    return signals