#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Choppiness Index regime filter + 1d Donchian breakout + volume confirmation
# Choppiness Index > 61.8 = range (mean revert), < 38.2 = trending (trend follow)
# Long: Donchian breakout above upper band in trending regime (1d ADX > 25)
# Short: Donchian breakdown below lower band in trending regime (1d ADX > 25)
# Exit: Opposite Donchian band touch or 1.5x ATR stop
# Designed to work in both bull (trend follow) and bear (trend follow) markets
# Target: 20-35 trades/year to avoid fee drag
name = "4h_Chop_Donchian1d_Volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for regime and Donchian (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d ADX for trend strength filter (trending when ADX > 25)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range
    tr1 = np.abs(high_1d - low_1d)
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    tr_1d[0] = high_1d[0] - low_1d[0]
    
    # Directional Movement
    up_move = np.diff(high_1d, prepend=high_1d[0])
    down_move = -np.diff(low_1d, prepend=low_1d[0])
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    # Smoothed values
    tr_14 = pd.Series(tr_1d).ewm(span=14, adjust=False).mean().values
    plus_di_14 = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values / tr_14
    minus_di_14 = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values / tr_14
    dx = 100 * np.abs(plus_di_14 - minus_di_14) / (plus_di_14 + minus_di_14 + 1e-10)
    adx_1d = pd.Series(dx).ewm(span=14, adjust=False).mean().values
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    
    # 1d Donchian channels (20-period)
    high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donch_high_1d = align_htf_to_ltf(prices, df_1d, high_20)
    donch_low_1d = align_htf_to_ltf(prices, df_1d, low_20)
    
    # 4h Choppiness Index (14-period)
    tr_4h = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr_4h[0] = high[0] - low[0]
    atr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    sum_tr_14 = pd.Series(tr_4h).rolling(window=14, min_periods=14).sum().values
    highest_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_tr_14 / 14) / np.log10(highest_high_14 - lowest_low_14 + 1e-10)
    chop = np.where((highest_high_14 - lowest_low_14) > 0, chop, 50.0)  # avoid division by zero
    
    # 4h ATR for stops
    atr_4h = pd.Series(tr_4h).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        if np.isnan(adx_1d_aligned[i]) or np.isnan(donch_high_1d[i]) or \
           np.isnan(donch_low_1d[i]) or np.isnan(chop[i]) or np.isnan(atr_4h[i]):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x average volume (20-period)
        if i >= 20:
            avg_volume = np.mean(volume[i-20:i])
        else:
            avg_volume = volume[i]
        volume_filter = volume[i] > 1.5 * avg_volume
        
        # Regime filter: trending when ADX > 25
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: Donchian breakout above upper band in trending regime + volume
            if trending and close[i] > donch_high_1d[i] and volume_filter:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakdown below lower band in trending regime + volume
            elif trending and close[i] < donch_low_1d[i] and volume_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: Touch lower Donchian band or 1.5x ATR stop
            if close[i] < donch_low_1d[i] or close[i] < high[i-1] - 1.5 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: Touch upper Donchian band or 1.5x ATR stop
            if close[i] > donch_high_1d[i] or close[i] > low[i-1] + 1.5 * atr_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals