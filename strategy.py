#!/usr/bin/env python3
"""
12h_Donchian20_Breakout_1wTrendFilter_VolumeSpike_v1
Hypothesis: Trade 12h Donchian(20) breakouts aligned with weekly EMA50 trend and volume spike (>2.0*ATR14).
Uses discrete sizing 0.25 to limit fee drag. Target: 12-37 trades/year to avoid fee drag while maintaining edge.
Weekly trend filter ensures we only trade with the higher timeframe momentum, reducing whipsaws in both bull and bear markets.
Volume confirmation adds conviction to breakouts. ATR-based stoploss via signal=0 when price violates opposite Donchian band.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate ATR14 for volume confirmation and stoploss
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian(20) channels
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for weekly EMA50, ATR, and Donchian channels
    start_idx = max(50, 14, lookback)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 2.0 * ATR
        volume_confirm = volume[i] > 2.0 * atr[i]
        
        # Determine weekly trend from EMA50
        weekly_close_aligned = align_htf_to_ltf(prices, df_1w, close_1w)[i]
        if np.isnan(weekly_close_aligned):
            signals[i] = 0.0
            continue
            
        if weekly_close_aligned > ema_50_1w_aligned[i]:
            weekly_trend = 'bullish'  # only allow longs
        elif weekly_close_aligned < ema_50_1w_aligned[i]:
            weekly_trend = 'bearish'  # only allow shorts
        else:
            weekly_trend = 'neutral'  # no trades in neutral zone
        
        if position == 0:
            # Long setup: price breaks above Donchian upper band AND volume confirm AND bullish weekly trend
            long_setup = (close[i] > highest_high[i]) and volume_confirm and (weekly_trend == 'bullish')
            
            # Short setup: price breaks below Donchian lower band AND volume confirm AND bearish weekly trend
            short_setup = (close[i] < lowest_low[i]) and volume_confirm and (weekly_trend == 'bearish')
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: price breaks below Donchian lower band OR weekly trend turns bearish
            if (close[i] < lowest_low[i]) or (weekly_trend == 'bearish'):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: price breaks above Donchian upper band OR weekly trend turns bullish
            if (close[i] > highest_high[i]) or (weekly_trend == 'bullish'):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian20_Breakout_1wTrendFilter_VolumeSpike_v1"
timeframe = "12h"
leverage = 1.0