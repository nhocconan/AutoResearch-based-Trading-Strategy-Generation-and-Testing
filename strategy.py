#!/usr/bin/env python3
"""
Hypothesis: 4h timeframe with 1d HMA(21) trend filter + Donchian(20) breakout + volume confirmation.
Long when price breaks above 20-period high with 1d HMA > previous HMA (uptrend) and volume > 1.5x 20-period volume average.
Short when price breaks below 20-period low with 1d HMA < previous HMA (downtrend) and volume > 1.5x 20-period volume average.
Uses HMA for smooth trend detection and Donchian for structure breakouts, designed to work in both bull and bear markets by following the trend direction from higher timeframe.
Target trades: 20-50/year to avoid fee drag while capturing significant moves.
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
    
    # Get 1d data for HMA trend filter and volume average
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d HMA(21) - Hull Moving Average
    def hull_moving_avg(values, window):
        if len(values) < window:
            return np.full_like(values, np.nan)
        half_window = window // 2
        sqrt_window = int(np.sqrt(window))
        
        # WMA of half period
        weights_half = np.arange(1, half_window + 1)
        wma_half = np.convolve(values, weights_half, mode='valid') / weights_half.sum()
        wma_half = np.concatenate([np.full(half_window-1, np.nan), wma_half])
        
        # WMA of full period
        weights_full = np.arange(1, window + 1)
        wma_full = np.convolve(values, weights_full, mode='valid') / weights_full.sum()
        wma_full = np.concatenate([np.full(window-1, np.nan), wma_full])
        
        # Raw HMA: 2*WMA(half) - WMA(full)
        raw_hma = 2 * wma_half - wma_full
        
        # Final WMA of raw_hma with sqrt period
        if len(raw_hma) < sqrt_window:
            return np.full_like(values, np.nan)
        weights_sqrt = np.arange(1, sqrt_window + 1)
        wma_sqrt = np.convolve(raw_hma, weights_sqrt, mode='valid') / weights_sqrt.sum()
        wma_sqrt = np.concatenate([np.full(sqrt_window-1, np.nan), wma_sqrt])
        
        return wma_sqrt
    
    hma_21_1d = hull_moving_avg(close_1d, 21)
    
    # Calculate 4h Donchian(20) channels
    def donchian_channel(high_vals, low_vals, window):
        upper = pd.Series(high_vals).rolling(window=window, min_periods=window).max().values
        lower = pd.Series(low_vals).rolling(window=window, min_periods=window).min().values
        return upper, lower
    
    donchian_upper, donchian_lower = donchian_channel(high, low, 20)
    
    # Calculate 4h volume 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align all to primary timeframe (4h)
    hma_21_1d_aligned = align_htf_to_ltf(prices, df_1d, hma_21_1d)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_1d, donchian_upper)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_1d, donchian_lower)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 50  # need enough for HMA and Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(hma_21_1d_aligned[i]) or 
            np.isnan(donchian_upper_aligned[i]) or 
            np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current 4h volume > 1.5x 20-period average
        volume_confirmed = volume[i] > 1.5 * vol_ma_20_aligned[i]
        
        if position == 0:
            # Long: price breaks above 20-period high with 1d HMA uptrend and volume
            if (close[i] > donchian_upper_aligned[i] and 
                hma_21_1d_aligned[i] > hma_21_1d_aligned[i-1] and 
                volume_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 20-period low with 1d HMA downtrend and volume
            elif (close[i] < donchian_lower_aligned[i] and 
                  hma_21_1d_aligned[i] < hma_21_1d_aligned[i-1] and 
                  volume_confirmed):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price falls back below 20-period low (opposite side of channel)
            if close[i] < donchian_lower_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price rises back above 20-period high (opposite side of channel)
            if close[i] > donchian_upper_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_1dHMA21_Donchian20_Breakout_Volume_Confirm"
timeframe = "4h"
leverage = 1.0