#!/usr/bin/env python3
"""
Hypothesis: 1d timeframe strategy using 1w Donchian breakout with 1d EMA50 trend filter and volume spike confirmation.
- Uses 1w for signal direction (Donchian breakout) and 1d for trend filter (EMA50)
- 1d only for execution to minimize fee drag
- Volume confirmation: > 2.0x 20-period average
- Position size: 0.25 (discrete level to balance return and fee drag)
- Target: 10-25 trades/year (40-100 over 4 years) to avoid fee drag
- Works in bull/bear via trend filter and volume confirmation
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
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1w Donchian channels for trend direction (HTF = 1w)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    donch_hi_1w = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    donch_lo_1w = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (use prior completed 1w bar)
    donch_hi_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_hi_1w)
    donch_lo_1w_aligned = align_htf_to_ltf(prices, df_1w, donch_lo_1w)
    
    # 1d EMA50 for trend filter (HTF = 1d)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 20)  # EMA50, volume MA, Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donch_hi_1w_aligned[i]) or
            np.isnan(donch_lo_1w_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Donchian breakout signals (using current close vs prior levels)
        breakout_up = close[i] > donch_hi_1w_aligned[i-1]  # Close above prior 1w Donchian high
        breakout_down = close[i] < donch_lo_1w_aligned[i-1]  # Close below prior 1w Donchian low
        
        if position == 0:
            # Long: 1w Donchian breakout up AND price > 1d EMA50 AND volume confirmation
            if breakout_up and volume_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: 1w Donchian breakout down AND price < 1d EMA50 AND volume confirmation
            elif breakout_down and volume_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1w Donchian breakout down OR price < 1d EMA50 (trend flip)
            if breakout_down or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 1w Donchian breakout up OR price > 1d EMA50 (trend flip)
            if breakout_up or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_VolumeSpike"
timeframe = "1d"
leverage = 1.0