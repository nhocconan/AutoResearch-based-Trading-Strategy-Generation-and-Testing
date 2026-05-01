#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Uses Donchian channels from 1d HTF for structural breakouts (wider timeframe = cleaner signals),
# 1d EMA34 for trend alignment, and volume spike (>2.0x 24-bar MA) for confirmation.
# Designed for 6h timeframe to achieve 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25).
# Works in both bull and bear markets via trend filter and tight entry conditions.

name = "6h_Donchian20_Breakout_1dEMA34_Trend_VolumeConfirm_v1"
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
    
    # Calculate 1d EMA34 for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 1d Donchian(20) channels for breakout
    # Based on previous 20 1d bars (shifted by 1 to avoid look-ahead)
    donchian_high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    donchian_low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    
    # Volume confirmation: current volume > 2.0 * 24-period average volume on 6h
    volume_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_confirm = volume > (volume_ma_24 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20, 24) + 1  # 35 (for EMA34, Donchian20, and volume MA)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(donchian_high_20_aligned[i]) or
            np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(volume_ma_24[i])):
            signals[i] = 0.0
            if position != 0:
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Trend filter: 1d EMA34 direction
        uptrend = curr_close > ema_34_1d_aligned[i]
        downtrend = curr_close < ema_34_1d_aligned[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # 6h Donchian(20) breakout conditions (using 1d levels)
        breakout_high = curr_high > donchian_high_20_aligned[i]  # Break above 1d Donchian high
        breakdown_low = curr_low < donchian_low_20_aligned[i]   # Break below 1d Donchian low
        
        if position == 0:  # Flat - look for new entries
            # Long: 1d Donchian high breakout AND uptrend AND volume confirmation
            if breakout_high and uptrend and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 1d Donchian low breakdown AND downtrend AND volume confirmation
            elif breakdown_low and downtrend and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on 1d Donchian low breakdown (reversal signal)
            if curr_low < donchian_low_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on 1d Donchian high breakout (reversal signal)
            if curr_high > donchian_high_20_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals