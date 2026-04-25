#!/usr/bin/env python3
"""
12h_KAMA_Regime_Donchian20_Exit
Hypothesis: Trade 12h timeframe using KAMA(10) for adaptive trend direction, 
combined with choppiness regime filter (CHOP > 61.8 = range, CHOP < 38.2 = trend) 
to avoid whipsaws, and weekly Donchian(20) for structured exits. 
In trending regimes (CHOP < 38.2): follow KAMA direction (long if price > KAMA, short if price < KAMA). 
In ranging regimes (CHOP > 61.8): mean revert at weekly Donchian bands (long at lower band, short at upper band). 
Volume confirmation (1d volume > 1.5x 20-bar MA) required for all entries. 
Uses discrete sizing 0.25 to control fees and drawdown. 
Designed to work in both bull and bear markets via regime adaptation.
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
    
    # Get 1d data for volume confirmation and choppiness calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d choppiness indicator (14-period)
    def choppiness(high_arr, low_arr, close_arr, period=14):
        atr_sum = np.zeros_like(close_arr)
        true_range = np.maximum(high_arr - low_arr, 
                               np.maximum(np.abs(high_arr - np.roll(close_arr, 1)), 
                                          np.abs(low_arr - np.roll(close_arr, 1))))
        true_range[0] = high_arr[0] - low_arr[0]  # first TR
        for i in range(1, len(true_range)):
            atr_sum[i] = atr_sum[i-1] + true_range[i]
        
        # Avoid division by zero
        max_high = np.maximum.accumulate(high_arr)
        min_low = np.minimum.accumulate(low_arr)
        range_hl = max_high - min_low
        
        chop = np.full_like(close_arr, 50.0, dtype=float)  # default to neutral
        for i in range(period, len(close_arr)):
            if range_hl[i] > 0 and atr_sum[i] > 0:
                chop[i] = 100 * np.log10(atr_sum[i] / range_hl[i]) / np.log10(period)
        return chop
    
    chop_1d = choppiness(high_1d, low_1d, close_1d, 14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Calculate 1d volume spike confirmation (volume > 1.5x 20-bar MA)
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (1.5 * vol_ma_1d)
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d)
    
    # Get weekly data for Donchian channel (20-period) - used for exits and range signals
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    donchian_high_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donchian_low_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    donchian_high_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_high_1w)
    donchian_low_1w_aligned = align_htf_to_ltf(prices, df_1w, donchian_low_1w)
    
    # Calculate KAMA(10) on 1d close for adaptive trend
    def kama(close, period=10, fast=2, slow=30):
        # Efficiency ratio
        change = np.abs(np.diff(close, period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close.shape) > 1 else \
                     np.sum(np.abs(np.diff(close)), axis=0)
        # Handle 1D case
        volatility = np.array([np.sum(np.abs(np.diff(close[i:i+period]))) 
                              for i in range(len(change))])
        er = np.zeros_like(close)
        er[period:] = change / (volatility + 1e-10)
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # Initialize KAMA
        kama_vals = np.full_like(close, np.nan, dtype=float)
        kama_vals[period] = close[period]
        
        for i in range(period+1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        
        return kama_vals
    
    kama_1d = kama(close_1d, 10, 2, 30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (10), choppiness (14), volume MA (20), Donchian (20)
    start_idx = max(30, 20)  # 30 covers KAMA warmup + buffer
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(chop_1d_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i]) or np.isnan(donchian_high_1w_aligned[i]) or
            np.isnan(donchian_low_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Determine regime: trending (CHOP < 38.2) or ranging (CHOP > 61.8)
            if chop_1d_aligned[i] < 38.2:  # Trending regime
                # Follow KAMA direction
                long_setup = (close[i] > kama_1d_aligned[i]) and volume_spike_1d_aligned[i]
                short_setup = (close[i] < kama_1d_aligned[i]) and volume_spike_1d_aligned[i]
            elif chop_1d_aligned[i] > 61.8:  # Ranging regime
                # Mean revert at weekly Donchian bands
                long_setup = (close[i] <= donchian_low_1w_aligned[i]) and volume_spike_1d_aligned[i]
                short_setup = (close[i] >= donchian_high_1w_aligned[i]) and volume_spike_1d_aligned[i]
            else:  # Choppy transition zone - no trades
                long_setup = False
                short_setup = False
            
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
            # Exit conditions
            if chop_1d_aligned[i] < 38.2:  # Trending: exit if price < KAMA
                if close[i] < kama_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
            else:  # Ranging or transition: exit at opposite Donchian band
                if close[i] >= donchian_high_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit conditions
            if chop_1d_aligned[i] < 38.2:  # Trending: exit if price > KAMA
                if close[i] > kama_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
            else:  # Ranging or transition: exit at opposite Donchian band
                if close[i] <= donchian_low_1w_aligned[i]:
                    signals[i] = 0.0
                    position = 0
    
    return signals

name = "12h_KAMA_Regime_Donchian20_Exit"
timeframe = "12h"
leverage = 1.0