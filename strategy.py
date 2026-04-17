#!/usr/bin/env python3
"""
Hypothesis: 12h timeframe with 1d Donchian channel breakout and volume confirmation.
Trade breakouts of daily Donchian(20) levels with volume spike (>1.5x 20-period average).
Use 1w ADX > 25 to filter for trending markets and avoid ranging whipsaws.
In trending markets: buy breakouts above upper Donchian, sell breakdowns below lower Donchian.
Position sizing: 0.25 for entries, 0 for exits.
Target: 50-150 total trades over 4 years (12-37/year).
Donchian channels provide clear structure levels that work in both trending and ranging markets when filtered by ADX.
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
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Donchian channels (20-period)
    lookback = 20
    upper_donchian = pd.Series(high_1d).rolling(window=lookback, min_periods=lookback).max().values
    lower_donchian = pd.Series(low_1d).rolling(window=lookback, min_periods=lookback).min().values
    
    # Get 1w data for ADX filter
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate ADX (14)
    plus_dm = np.where((high_1w[1:] - high_1w[:-1]) > (low_1w[:-1] - low_1w[1:]), 
                       np.maximum(high_1w[1:] - high_1w[:-1], 0), 0)
    minus_dm = np.where((low_1w[:-1] - low_1w[1:]) > (high_1w[1:] - high_1w[:-1]), 
                        np.maximum(low_1w[:-1] - low_1w[1:], 0), 0)
    plus_dm = np.concatenate([[0], plus_dm])
    minus_dm = np.concatenate([[0], minus_dm])
    
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr3 = np.abs(low_1w - np.concatenate([[close_1w[0]], close_1w[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Volume filter: 1.5x 20-period average
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 12h
    upper_donchian_aligned = align_htf_to_ltf(prices, df_1d, upper_donchian)
    lower_donchian_aligned = align_htf_to_ltf(prices, df_1d, lower_donchian)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_donchian_aligned[i]) or np.isnan(lower_donchian_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Determine trend direction from ADX components
        plus_di = 100 * pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values / atr
        
        plus_di_aligned = align_htf_to_ltf(prices, df_1w, plus_di)
        minus_di_aligned = align_htf_to_ltf(prices, df_1w, minus_di)
        
        if np.isnan(plus_di_aligned[i]) or np.isnan(minus_di_aligned[i]):
            signals[i] = 0.0
            continue
            
        uptrend = plus_di_aligned[i] > minus_di_aligned[i]
        downtrend = plus_di_aligned[i] < minus_di_aligned[i]
        strong_trend = adx_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above upper Donchian, volume spike, strong trend
            if (close[i] > upper_donchian_aligned[i] and 
                volume[i] > vol_ma_20_aligned[i] * 1.5 and 
                strong_trend and uptrend):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian, volume spike, strong trend
            elif (close[i] < lower_donchian_aligned[i] and 
                  volume[i] > vol_ma_20_aligned[i] * 1.5 and 
                  strong_trend and downtrend):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price returns below lower Donchian or trend weakens
            if close[i] < lower_donchian_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price returns above upper Donchian or trend weakens
            if close[i] > upper_donchian_aligned[i] or adx_aligned[i] < 20:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_1dDonchian20_Volume_ADX"
timeframe = "12h"
leverage = 1.0