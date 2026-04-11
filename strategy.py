#!/usr/bin/env python3
# 4h_1d_adx_donchian_breakout_v1
# Strategy: 4h Donchian(20) breakout with ADX trend strength filter and volume confirmation
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: Donchian breakouts capture breakout momentum, while ADX > 25 ensures we only trade in strong trends.
# Volume confirmation avoids false breakouts. Works in both bull (long breakouts) and bear (short breakouts).
# Low frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_adx_donchian_breakout_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ADX(14) for trend strength filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate ADX components
    plus_dm = np.where((high_1d[1:] - high_1d[:-1]) > (low_1d[:-1] - low_1d[1:]), 
                       np.maximum(high_1d[1:] - high_1d[:-1], 0), 0)
    minus_dm = np.where((low_1d[:-1] - low_1d[1:]) > (high_1d[1:] - high_1d[:-1]), 
                        np.maximum(low_1d[:-1] - low_1d[1:], 0), 0)
    tr = np.maximum(high_1d[1:] - low_1d[1:], 
                    np.maximum(np.abs(high_1d[1:] - close_1d[:-1]), 
                               np.abs(low_1d[1:] - close_1d[:-1])))
    
    # Add first element as 0 for alignment
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    tr = np.concatenate([[0], tr])
    
    # Smooth with Wilder's smoothing (alpha = 1/period)
    atr = np.zeros_like(tr)
    plus_di = np.zeros_like(tr)
    minus_di = np.zeros_like(tr)
    dx = np.zeros_like(tr)
    adx = np.zeros_like(tr)
    
    atr[0] = tr[0]
    plus_di[0] = 0
    minus_di[0] = 0
    
    for i in range(1, len(tr)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
        plus_di[i] = (plus_di[i-1] * 13 + plus_dm[i]) / (atr[i] * 100) if atr[i] != 0 else 0
        minus_di[i] = (minus_di[i-1] * 13 + minus_dm[i]) / (atr[i] * 100) if atr[i] != 0 else 0
        dx[i] = (np.abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i]) * 100) if (plus_di[i] + minus_di[i]) != 0 else 0
        adx[i] = (adx[i-1] * 13 + dx[i]) / 14 if i >= 14 else dx[i]
    
    # Pad beginning with NaN for first 14 periods
    adx_full = np.full_like(tr, np.nan)
    adx_full[14:] = adx[14:]
    adx_1d = adx_full
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(lookback, n):
        # Skip if any required data is invalid
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(adx_1d_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Trend filter: ADX > 25 indicates strong trend
        strong_trend = adx_1d_aligned[i] > 25
        
        # Entry logic: Donchian breakout + ADX + volume
        if (close[i] > highest_high[i] and strong_trend and vol_confirm[i] and position != 1):
            position = 1
            signals[i] = 0.25
        elif (close[i] < lowest_low[i] and strong_trend and vol_confirm[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: trend weakening or opposite breakout
        elif position == 1 and (not strong_trend or close[i] < lowest_low[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not strong_trend or close[i] > highest_high[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals