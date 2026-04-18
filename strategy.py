#!/usr/bin/env python3
"""
12h 48-bar Donchian Breakout with Volume Spike and ADX Trend Filter
Strategy: Enter long when price breaks above 48-bar (24-day) Donchian high with volume spike and ADX > 25,
          short when price breaks below 48-bar Donchian low with volume spike and ADX > 25.
          Exit on opposite Donchian break or ADX drop below 20.
          Designed for 12h timeframe to capture multi-day trends with volume confirmation.
          Uses daily ADX for trend strength to avoid choppy markets.
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
    
    # Get daily data for ADX trend filter (once before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily ADX (14-period)
    # True Range
    tr1 = df_1d['high'] - df_1d['low']
    tr2 = abs(df_1d['high'] - df_1d['close'].shift(1))
    tr3 = abs(df_1d['low'] - df_1d['close'].shift(1))
    tr = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    
    # Directional Movement
    up_move = df_1d['high'] - df_1d['high'].shift(1)
    down_move = df_1d['low'].shift(1) - df_1d['low']
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed values
    atr = pd.Series(tr).ewm(alpha=1/14, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/14, adjust=False).mean().values / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False).mean().values
    
    # Align daily ADX to 12h timeframe
    adx_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # 48-bar Donchian channels (24 days at 12h = 48 periods)
    lookback = 48
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume spike detection (2x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = max(100, lookback)  # need enough history
    
    for i in range(start_idx, n):
        if (np.isnan(adx_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        donchian_high = highest_high[i]
        donchian_low = lowest_low[i]
        adx_val = adx_aligned[i]
        
        if position == 0:
            # Long: break above Donchian high with volume spike and strong trend (ADX > 25)
            if (price > donchian_high and volume_spike[i] and adx_val > 25):
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume spike and strong trend (ADX > 25)
            elif (price < donchian_low and volume_spike[i] and adx_val > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long position management
            signals[i] = 0.25
            # Exit: price breaks below Donchian low OR trend weakens (ADX < 20)
            if price < donchian_low or adx_val < 20:
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position management
            signals[i] = -0.25
            # Exit: price breaks above Donchian high OR trend weakens (ADX < 20)
            if price > donchian_high or adx_val < 20:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "12h_Donchian48_VolumeSpike_ADXFilter"
timeframe = "12h"
leverage = 1.0