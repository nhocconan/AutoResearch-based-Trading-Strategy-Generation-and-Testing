#!/usr/bin/env python3
"""
Hypothesis: 6h timeframe with weekly Donchian channel breakout + volume spike + 1d ADX trend filter.
Long when price breaks above weekly Donchian high (20) with volume > 2x 20-period average and 1d ADX > 25.
Short when price breaks below weekly Donchian low (20) with volume > 2x 20-period average and 1d ADX > 25.
Weekly Donchian captures major structural breaks; volume spike confirms institutional interest; ADX filter ensures trending market.
Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag. Uses discrete sizing 0.25.
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
    
    # Get weekly data for Donchian channel
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly Donchian channel (20-period)
    def rolling_max(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).max().values
    
    def rolling_min(arr, window):
        return pd.Series(arr).rolling(window=window, min_periods=window).min().values
    
    donchian_high_1w = rolling_max(high_1w, 20)
    donchian_low_1w = rolling_min(low_1w, 20)
    
    # Get 1d data for ADX trend filter
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ADX (14-period)
    def calculate_adx(high, low, close, window=14):
        plus_dm = np.zeros_like(high)
        minus_dm = np.zeros_like(high)
        tr = np.zeros_like(high)
        
        for i in range(1, len(high)):
            plus_dm[i] = max(high[i] - high[i-1], 0)
            minus_dm[i] = max(low[i-1] - low[i], 0)
            if plus_dm[i] == minus_dm[i]:
                plus_dm[i] = 0
                minus_dm[i] = 0
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        # Wilder's smoothing
        atr = np.zeros_like(high)
        atr[window] = np.mean(tr[1:window+1])
        for i in range(window+1, len(high)):
            atr[i] = (atr[i-1] * (window-1) + tr[i]) / window
        
        plus_di = np.zeros_like(high)
        minus_di = np.zeros_like(high)
        dx = np.zeros_like(high)
        
        # Smooth +DM and -DM
        plus_dm_smooth = np.zeros_like(high)
        minus_dm_smooth = np.zeros_like(high)
        plus_dm_smooth[window] = np.sum(plus_dm[1:window+1])
        minus_dm_smooth[window] = np.sum(minus_dm[1:window+1])
        
        for i in range(window+1, len(high)):
            plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (window-1) + plus_dm[i]) / window
            minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (window-1) + minus_dm[i]) / window
        
        # Calculate +DI and -DI
        for i in range(window, len(high)):
            if atr[i] != 0:
                plus_di[i] = (plus_dm_smooth[i] / atr[i]) * 100
                minus_di[i] = (minus_dm_smooth[i] / atr[i]) * 100
                if plus_di[i] + minus_di[i] != 0:
                    dx[i] = (abs(plus_di[i] - minus_di[i]) / (plus_di[i] + minus_di[i])) * 100
        
        # Calculate ADX (smoothed DX)
        adx = np.zeros_like(high)
        adx[2*window] = np.mean(dx[window:2*window+1])
        for i in range(2*window+1, len(high)):
            adx[i] = (adx[i-1] * (window-1) + dx[i]) / window
        
        return adx
    
    adx_1d = calculate_adx(high_1d, low_1d, close_1d, 14)
    
    # Calculate 1d volume 20-period average
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all to 6h
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx_1d)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 200  # need enough for ADX calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_1w_aligned[i]) or np.isnan(donchian_low_1w_aligned[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_ma_20_1d_aligned[i]) or 
            np.isnan(volume_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 2x 20-period average
        volume_confirmed = volume_1d_aligned[i] > 2.0 * vol_ma_20_1d_aligned[i]
        
        # ADX trend filter: ADX > 25 indicates trending market
        trending = adx_1d_aligned[i] > 25
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume and trend
            if (close[i] > donchian_high_1w_aligned[i] and 
                volume_confirmed and 
                trending):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume and trend
            elif (close[i] < donchian_low_1w_aligned[i] and 
                  volume_confirmed and 
                  trending):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly Donchian low or trend weakens
            if (close[i] < donchian_low_1w_aligned[i] or 
                adx_1d_aligned[i] < 20):  # ADX < 20 indicates ranging/weak trend
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly Donchian high or trend weakens
            if (close[i] > donchian_high_1w_aligned[i] or 
                adx_1d_aligned[i] < 20):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1wDonchian20_Volume_ADX"
timeframe = "6h"
leverage = 1.0