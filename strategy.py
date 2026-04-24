#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d ADX regime filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1d for ADX trend strength.
- ADX > 25 indicates trending market (breakout strategy), ADX < 20 indicates ranging (mean reversion at Camarilla H3/L3).
- Entry: Long when price breaks above Camarilla R3 AND ADX > 25 (bullish breakout in trend).
         Short when price breaks below Camarilla S3 AND ADX > 25 (bearish breakout in trend).
         In ranging (ADX < 20): Long when price touches Camarilla S3 AND reverses up (close > low).
                                Short when price touches Camarilla R3 AND reverses down (close < high).
- Exit: Opposite Camarilla breakout or ADX regime shift to ranging.
- Volume confirmation: current volume > 1.3 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ADX
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 1d
    # True Range
    tr1 = pd.Series(df_1d['high']).diff().abs()
    tr2 = (pd.Series(df_1d['high']) - pd.Series(df_1d['low'].shift())).abs()
    tr3 = (pd.Series(df_1d['low']) - pd.Series(df_1d['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1d['high']).diff()
    down_move = -pd.Series(df_1d['low']).diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed DM
    plus_dm_smooth = pd.Series(plus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    minus_dm_smooth = pd.Series(minus_dm).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Indicators
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # DX and ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Align 1d ADX to 4h
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Camarilla levels from previous 1d bar (for current 4h bar)
    # We need to get the previous completed 1d bar's OHLC for each 4h bar
    # align_htf_to_ltf will give us the previous day's values aligned to each 4h bar
    df_1d_prev = df_1d.copy()
    # Shift OHLC by 1 to get previous day's values (for calculating today's Camarilla levels)
    df_1d_prev['high'] = df_1d_prev['high'].shift(1)
    df_1d_prev['low'] = df_1d_prev['low'].shift(1)
    df_1d_prev['close'] = df_1d_prev['close'].shift(1)
    
    # Calculate Camarilla levels using previous day's OHLC
    high_prev = df_1d_prev['high'].values
    low_prev = df_1d_prev['low'].values
    close_prev = df_1d_prev['close'].values
    
    # Camarilla levels
    R3 = close_prev + (high_prev - low_prev) * 1.1 / 4
    S3 = close_prev - (high_prev - low_prev) * 1.1 / 4
    R4 = close_prev + (high_prev - low_prev) * 1.1 / 2
    S4 = close_prev - (high_prev - low_prev) * 1.1 / 2
    
    # Align Camarilla levels to 4h
    R3_aligned = align_htf_to_ltf(prices, df_1d_prev, R3)
    S3_aligned = align_htf_to_ltf(prices, df_1d_prev, S3)
    R4_aligned = align_htf_to_ltf(prices, df_1d_prev, R4)
    S4_aligned = align_htf_to_ltf(prices, df_1d_prev, S4)
    
    # Volume confirmation: current volume > 1.3 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, 20)  # Need enough 1d bars for ADX and Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        adx_val = adx_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if adx_val > 25:  # Trending regime: breakout strategy
                    # Bullish breakout: price closes above Camarilla R3
                    if curr_close > R3_aligned[i]:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price closes below Camarilla S3
                    elif curr_close < S3_aligned[i]:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging regime (ADX < 20): mean reversion at extremes
                    # Long when price touches S3 and shows reversal (close > low)
                    if curr_low <= S3_aligned[i] and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches R3 and shows reversal (close < high)
                    elif curr_high >= R3_aligned[i] and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below Camarilla S3 OR ADX drops to ranging
            if curr_close < S3_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Camarilla R3 OR ADX drops to ranging
            if curr_close > R3_aligned[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3S3_1dADXRegime_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0