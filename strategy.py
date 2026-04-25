#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_ATRTrail_VolumeSpike_v1
Hypothesis: Trade Donchian(20) breakouts on 4h with volume spike confirmation and ATR-based trailing stoploss. Uses 1d EMA50 as trend filter to avoid counter-trend whipsaw in bear markets. Volume spike confirms breakout strength. ATR trailing stop limits downside during reversals. Discrete sizing (0.30) controls fee drag. Designed to work in both bull (trend continuation) and bear (mean reversion after spikes) markets by combining breakout logic with trend filter and volatility-based exit.
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for HTF trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Calculate ATR (14) for volatility-based stop
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume spike: current volume > 2.0 * 20-period volume MA
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_since_entry = 0.0  # for long ATR trail
    lowest_since_entry = 0.0   # for short ATR trail
    
    # Start index: need warmup for Donchian (20), ATR (14), EMA50 (50), volume MA (20)
    start_idx = max(lookback, 14, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0 if position == 0 else (0.30 if position == 1 else -0.30)
            continue
        
        # Determine 1d HTF trend (bullish = price above EMA50)
        htf_1d_bullish = close[i] > ema_50_1d_aligned[i]
        htf_1d_bearish = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long setup: price breaks above Donchian high + 1d uptrend + volume spike
            long_setup = (close[i] > highest_high[i]) and htf_1d_bullish and volume_spike[i]
            
            # Short setup: price breaks below Donchian low + 1d downtrend + volume spike
            short_setup = (close[i] < lowest_low[i]) and htf_1d_bearish and volume_spike[i]
            
            if long_setup:
                signals[i] = 0.30
                position = 1
                highest_since_entry = close[i]
            elif short_setup:
                signals[i] = -0.30
                position = -1
                lowest_since_entry = close[i]
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.30
            # Update highest close since entry
            if close[i] > highest_since_entry:
                highest_since_entry = close[i]
            # ATR trailing stop: exit if price drops 2.5*ATR from highest since entry
            if close[i] < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Optional: exit if 1d trend turns bearish (uncomment for tighter stops)
            # elif not htf_1d_bullish:
            #     signals[i] = 0.0
            #     position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.30
            # Update lowest close since entry
            if close[i] < lowest_since_entry:
                lowest_since_entry = close[i]
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest since entry
            if close[i] > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Optional: exit if 1d trend turns bullish (uncomment for tighter stops)
            # elif htf_1d_bullish:
            #     signals[i] = 0.0
            #     position = 0
    
    return signals

name = "4h_Donchian20_Breakout_ATRTrail_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0