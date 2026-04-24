#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w ADX regime filter and volume confirmation.
- Primary timeframe: 12h for execution, HTF: 1w for ADX trend strength.
- ADX > 25 indicates trending market (breakout strategy), ADX < 20 indicates ranging (mean reversion at Donchian mid).
- Entry: Long when price breaks above Donchian(20) upper AND ADX > 25 (bullish breakout in trend).
         Short when price breaks below Donchian(20) lower AND ADX > 25 (bearish breakout in trend).
         In ranging (ADX < 20): Long when price touches lower Donchian AND reverses up (close > low).
                                Short when price touches upper Donchian AND reverses down (close < high).
- Exit: Opposite Donchian breakout or ADX regime shift to ranging.
- Volume confirmation: current volume > 1.3 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.
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
    
    # Get 1w data for ADX
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Calculate ADX (14-period) on 1w
    # True Range
    tr1 = pd.Series(df_1w['high']).diff().abs()
    tr2 = (pd.Series(df_1w['high']) - pd.Series(df_1w['low'].shift())).abs()
    tr3 = (pd.Series(df_1w['low']) - pd.Series(df_1w['close'].shift())).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Directional Movement
    up_move = pd.Series(df_1w['high']).diff()
    down_move = -pd.Series(df_1w['low']).diff()
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
    
    # Align 1w ADX to 12h
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Donchian channels (20-period) on 12h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: current volume > 1.3 * 20-period volume MA (on 12h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(30, lookback, 20)  # Need enough 1w bars for ADX and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(adx_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
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
                    # Bullish breakout: price closes above upper Donchian
                    if curr_close > highest_high[i]:
                        signals[i] = 0.25
                        position = 1
                    # Bearish breakout: price closes below lower Donchian
                    elif curr_close < lowest_low[i]:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging regime (ADX < 20): mean reversion at extremes
                    # Long when price touches lower Donchian and shows reversal (close > low)
                    if curr_low <= lowest_low[i] and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches upper Donchian and shows reversal (close < high)
                    elif curr_high >= highest_high[i] and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below Donchian mid OR ADX drops to ranging
            if curr_close < donchian_mid[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian mid OR ADX drops to ranging
            if curr_close > donchian_mid[i] or adx_val < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1wADXRegime_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0