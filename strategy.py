#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with weekly Donchian channel breakout + volume confirmation + ADX trend filter.
Long when price breaks above weekly Donchian(20) high with volume > 1.5x 20-period average and weekly ADX > 25.
Short when price breaks below weekly Donchian(20) low with volume > 1.5x 20-period average and weekly ADX > 25.
Weekly Donchian captures major trend structure; breakouts with volume and trend filter reduce false signals.
Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag. Uses discrete sizing 0.25.
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
    
    # Calculate weekly Donchian(20)
    donchian_window = 20
    high_ma_20 = pd.Series(high_1w).rolling(window=donchian_window, min_periods=donchian_window).max().values
    low_ma_20 = pd.Series(low_1w).rolling(window=donchian_window, min_periods=donchian_window).min().values
    
    # Calculate weekly ADX(14)
    adx_window = 14
    tr1 = pd.Series(high_1w - low_1w).values
    tr2 = pd.Series(np.abs(high_1w - np.roll(close_1w, 1))).values
    tr3 = pd.Series(np.abs(low_1w - np.roll(close_1w, 1))).values
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First TR is just high-low
    
    atr = pd.Series(tr).rolling(window=adx_window, min_periods=adx_window).mean().values
    
    plus_dm = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    plus_di = 100 * pd.Series(plus_dm).rolling(window=adx_window, min_periods=adx_window).mean().values / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=adx_window, min_periods=adx_window).mean().values / atr
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = pd.Series(dx).rolling(window=adx_window, min_periods=adx_window).mean().values
    
    # Get 1d data for volume
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 1d
    high_ma_20_aligned = align_htf_to_ltf(prices, df_1w, high_ma_20)
    low_ma_20_aligned = align_htf_to_ltf(prices, df_1w, low_ma_20)
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for weekly calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(high_ma_20_aligned[i]) or np.isnan(low_ma_20_aligned[i]) or 
            np.isnan(adx_aligned[i]) or np.isnan(volume_1d_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume and trend (ADX > 25)
            if (close[i] > high_ma_20_aligned[i] and 
                volume_confirmed and 
                adx_aligned[i] > 25):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume and trend (ADX > 25)
            elif (close[i] < low_ma_20_aligned[i] and 
                  volume_confirmed and 
                  adx_aligned[i] > 25):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly Donchian low or trend weakens (ADX < 20)
            if (close[i] < low_ma_20_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly Donchian high or trend weakens (ADX < 20)
            if (close[i] > high_ma_20_aligned[i] or 
                adx_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wDonchian20_Volume_ADX"
timeframe = "1d"
leverage = 1.0