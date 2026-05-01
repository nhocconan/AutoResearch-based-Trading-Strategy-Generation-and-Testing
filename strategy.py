#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h HMA21 trend filter and volume confirmation.
# Uses Donchian channel for breakout structure, filtered by 12h HMA21 trend and volume > 1.5x 20-period median.
# Works in bull (buy breakouts with uptrend) and bear (sell breakdowns with downtrend).
# Discrete position sizing (0.25) to minimize fee churn. Target: 75-200 total trades over 4 years.

name = "4h_Donchian20_Breakout_12hHMA21_VolumeConfirm_v1"
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
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h HMA21 for trend filter
    def hma(arr, period):
        half = arr[::2]
        wma2 = np.convolve(half, np.arange(1, len(half)+1), 'valid') / np.arange(1, len(half)+1).sum()
        wma1 = np.convolve(arr, np.arange(1, len(arr)+1), 'valid') / np.arange(1, len(arr)+1).sum()
        return 2 * wma2[-len(wma1):] - wma1
    
    close_12h = df_12h['close'].values
    hma_21_12h = hma(close_12h, 21)
    hma_21_12h_aligned = align_htf_to_ltf(prices, df_12h, hma_21_12h)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Donchian(20) from previous period
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for HMA21 and volume median
    start_idx = 20
    
    for i in range(start_idx, n):
        if (np.isnan(hma_21_12h_aligned[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_volume = volume[i]
        
        # Trend filter: 12h HMA21 direction
        uptrend = curr_close > hma_21_12h_aligned[i]
        downtrend = curr_close < hma_21_12h_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Donchian breakout conditions
        breakout_up = curr_close > donchian_high[i]   # break above upper band
        breakout_down = curr_close < donchian_low[i]  # break below lower band
        
        if position == 0:  # Flat - look for new entries
            # Long: Breakout up AND uptrend AND volume confirmation
            if breakout_up and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Breakout down AND downtrend AND volume confirmation
            elif breakout_down and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on Donchian breakout down (reversal signal)
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout up (reversal signal)
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals