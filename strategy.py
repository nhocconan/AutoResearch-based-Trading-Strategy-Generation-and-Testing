#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) Breakout + Volume Spike + ADX Trend Filter
Donchian channel breakouts capture sustained momentum. Volume confirmation filters false breakouts.
ADX > 25 ensures we only trade in trending markets, reducing whipsaws in ranging conditions.
Discrete sizing 0.25 limits fee churn. Timeframe 4h targets 20-40 trades/year.
Works in both bull (breakouts up) and bear (breakdowns down) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate ADX(14) for trend filter on 1d timeframe
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need enough for ADX calculation
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    
    # Calculate +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm[0] = 0.0
    minus_dm[0] = 0.0
    
    # Smooth TR, +DM, -DM using Wilder's smoothing (alpha = 1/period)
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Calculate Donchian channels (20-period) on 4h data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 20)  # need Donchian, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(adx_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Close > upper Donchian (breakout) AND volume spike AND ADX > 25 (trending)
            if (close[i] > highest_high[i] and 
                volume[i] > 1.8 * vol_ma[i] and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: Close < lower Donchian (breakdown) AND volume spike AND ADX > 25 (trending)
            elif (close[i] < lowest_low[i] and 
                  volume[i] > 1.8 * vol_ma[i] and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Close back inside Donchian channel OR loss of trend (ADX < 20)
            exit_signal = False
            if position == 1:
                # Exit long when close < lower Donchian OR ADX < 20 (trend weakening)
                if close[i] < lowest_low[i] or adx_aligned[i] < 20:
                    exit_signal = True
            elif position == -1:
                # Exit short when close > upper Donchian OR ADX < 20 (trend weakening)
                if close[i] > highest_high[i] or adx_aligned[i] < 20:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian20_Breakout_VolumeSpike_ADXTrend"
timeframe = "4h"
leverage = 1.0