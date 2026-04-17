#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with weekly Donchian channel breakout + volume confirmation + ADX trend filter.
Long when price breaks above weekly Donchian high (20) with volume > 1.3x 20-period average and weekly ADX > 25.
Short when price breaks below weekly Donchian low (20) with volume > 1.3x 20-period average and weekly ADX > 25.
Weekly Donchian captures major swing levels; breakouts with volume and strong trend filter (ADX) reduce false signals in both bull and bear markets.
Target: 30-100 total trades over 4 years (7-25/year). Uses discrete sizing 0.25.
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
    
    # Get weekly data for Donchian and ADX
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channel (20)
    high_1w_series = pd.Series(high_1w)
    low_1w_series = pd.Series(low_1w)
    donch_high_20 = high_1w_series.rolling(window=20, min_periods=20).max().values
    donch_low_20 = low_1w_series.rolling(window=20, min_periods=20).min().values
    
    # Calculate weekly ADX (14)
    # +DM, -DM, TR
    up_move = high_1w_series.diff()
    down_move = low_1w_series.diff().multiply(-1)
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = high_1w_series - low_1w_series
    tr2 = abs(high_1w_series - close_1w_series.shift(1))
    tr3 = abs(low_1w_series - close_1w_series.shift(1))
    tr = pd.Series(np.maximum(tr1, np.maximum(tr2, tr3)))
    atr = tr.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    plus_di = 100 * (pd.Series(plus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).ewm(alpha=1/14, adjust=False, min_periods=14).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # Get 1d data for volume
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 1d
    donch_high_20_aligned = align_htf_to_ltf(prices, df_1w, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_1w, donch_low_20)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for ADX and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.3 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume and strong trend (ADX > 25)
            if (close[i] > donch_high_20_aligned[i] and 
                volume_confirmed and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume and strong trend (ADX > 25)
            elif (close[i] < donch_low_20_aligned[i] and 
                  volume_confirmed and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly Donchian low or trend weakens (ADX < 20)
            if (close[i] < donch_low_20_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly Donchian high or trend weakens (ADX < 20)
            if (close[i] > donch_high_20_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wDonchian20_Volume_ADX"
timeframe = "1d"
leverage = 1.0