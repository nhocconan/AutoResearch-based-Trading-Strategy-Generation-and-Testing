#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe with weekly Donchian channel breakout + volume confirmation + ATR filter.
Long when price breaks above weekly Donchian high (20) with volume > 1.5x 20-day average and ATR(14) < ATR(50) (low volatility breakout).
Short when price breaks below weekly Donchian low (20) with volume > 1.5x 20-day average and ATR(14) < ATR(50).
Weekly Donchian captures major structural levels; breakouts with volume and low volatility filter reduce false signals.
Target: 30-100 total trades over 4 years (7-25/year) to avoid fee drag. Uses discrete sizing 0.25.
Works in both bull (breakouts continue) and bear (breakdowns continue) markets via symmetric logic.
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
    
    # Calculate weekly Donchian channel (20 periods)
    def donchian_channel(high_arr, low_arr, window):
        high_max = pd.Series(high_arr).rolling(window=window, min_periods=window).max().values
        low_min = pd.Series(low_arr).rolling(window=window, min_periods=window).min().values
        return high_max, low_min
    
    donchian_high_20w, donchian_low_20w = donchian_channel(high_1w, low_1w, 20)
    
    # Get daily data for volume and ATR filters
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate daily volume 20-period average
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily ATR(14) and ATR(50) for volatility filter
    def atr(high_arr, low_arr, close_arr, window):
        tr1 = high_arr - low_arr
        tr2 = np.abs(high_arr - np.roll(close_arr, 1))
        tr3 = np.abs(low_arr - np.roll(close_arr, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # first TR is just high-low
        atr_vals = pd.Series(tr).ewm(span=window, adjust=False, min_periods=window).mean().values
        return atr_vals
    
    atr14_1d = atr(high_1d, low_1d, close_1d, 14)
    atr50_1d = atr(high_1d, low_1d, close_1d, 50)
    
    # Align all to 1d timeframe (primary)
    donchian_high_20w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_20w)
    donchian_low_20w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_20w)
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    volume_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    atr14_1d_aligned = align_htf_to_ltf(prices, df_1d, atr14_1d)
    atr50_1d_aligned = align_htf_to_ltf(prices, df_1d, atr50_1d)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for ATR50 and Donchian20
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donchian_high_20w_aligned[i]) or np.isnan(donchian_low_20w_aligned[i]) or 
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(volume_1d_aligned[i]) or 
            np.isnan(atr14_1d_aligned[i]) or np.isnan(atr50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 1d volume > 1.5x 20-day average
        volume_confirmed = volume_1d_aligned[i] > 1.5 * vol_ma_20_1d_aligned[i]
        # Volatility filter: ATR(14) < ATR(50) (low volatility breakout)
        vol_filter = atr14_1d_aligned[i] < atr50_1d_aligned[i]
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume and low volatility
            if (close[i] > donchian_high_20w_aligned[i] and 
                volume_confirmed and 
                vol_filter):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low with volume and low volatility
            elif (close[i] < donchian_low_20w_aligned[i] and 
                  volume_confirmed and 
                  vol_filter):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below weekly Donchian low or volatility increases
            if (close[i] < donchian_low_20w_aligned[i] or 
                atr14_1d_aligned[i] > atr50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above weekly Donchian high or volatility increases
            if (close[i] > donchian_high_20w_aligned[i] or 
                atr14_1d_aligned[i] > atr50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1wDonchian20_Volume_VolFilter"
timeframe = "1d"
leverage = 1.0