#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d trend filter (price > 1d SMA50 for long, < for short) and volume spike confirmation.
# Uses price channel breakouts aligned with daily trend and volume confirmation to capture momentum moves.
# Works in bull markets (buy breakouts with uptrend) and bear markets (sell breakdowns with downtrend).
# Target: 50-150 total trades over 4 years (12-37/year) on 6h timeframe.
# Discrete position sizing (0.25) to minimize fee churn.

name = "6h_Donchian20_Breakout_1dSMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d SMA50 for trend filter
    sma_50_1d = pd.Series(df_1d['close']).rolling(window=50, min_periods=50).mean().values
    sma_50_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_50_1d)
    
    # Calculate 20-period volume median for volume confirmation
    vol_median_20 = pd.Series(volume).rolling(window=20, min_periods=20).median().values
    
    # Calculate Donchian(20) channels from previous period OHLC
    donchian_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for SMA50, Donchian, and volume median
    start_idx = 50
    
    for i in range(start_idx, n):
        if (np.isnan(sma_50_1d_aligned[i]) or 
            np.isnan(vol_median_20[i]) or
            np.isnan(donchian_high_20[i]) or
            np.isnan(donchian_low_20[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d SMA50 direction
        uptrend = curr_close > sma_50_1d_aligned[i]
        downtrend = curr_close < sma_50_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-period volume median
        if vol_median_20[i] <= 0 or np.isnan(vol_median_20[i]):
            volume_confirm = False
        else:
            volume_confirm = curr_volume > (vol_median_20[i] * 1.5)
        
        # Donchian breakout conditions (using previous period channels)
        breakout_up = curr_close > donchian_high_20[i]   # break above upper channel
        breakout_down = curr_close < donchian_low_20[i]  # break below lower channel
        
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
            # Exit on Donchian breakdown (reversal signal)
            if curr_close < donchian_low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on Donchian breakout (reversal signal)
            if curr_close > donchian_high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals