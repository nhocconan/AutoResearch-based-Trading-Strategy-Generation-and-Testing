#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume spike (>2.0x average).
- Uses 1d for signal direction (Donchian breakout) and 1d for trend filter (EMA50)
- Volume spike reduces false breakouts (strict >2.0x to limit trades)
- No session filter to allow more opportunities in bear markets
- Position size: 0.25 (discrete level to minimize fee churn)
- Target: 12-37 trades/year (50-150 over 4 years) to avoid fee drag
- Works in bull/bear via trend filter and volume confirmation
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
    
    # Volume confirmation: > 2.0x 24-period average (strict for 12h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 1d Donchian channel (20-period)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian upper/lower bands (20-period)
    high_ma_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    low_ma_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (use prior completed 1d bar)
    donch_high_1d_aligned = align_htf_to_ltf(prices, df_1d, high_ma_20)
    donch_low_1d_aligned = align_htf_to_ltf(prices, df_1d, low_ma_20)
    
    # 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 24)  # EMA50, Donchian20, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donch_high_1d_aligned[i]) or
            np.isnan(donch_low_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Donchian breakout signals (using current close vs prior levels)
        breakout_up = close[i] > donch_high_1d_aligned[i-1]  # Close above prior 1d upper band
        breakout_down = close[i] < donch_low_1d_aligned[i-1]  # Close below prior 1d lower band
        
        if position == 0:
            # Long: 1d Donchian upper breakout up AND price > 1d EMA50 AND volume confirmation
            if breakout_up and volume_confirm and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: 1d Donchian lower breakout down AND price < 1d EMA50 AND volume confirmation
            elif breakout_down and volume_confirm and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1d Donchian lower break down OR price < 1d EMA50 (trend flip)
            if close[i] < donch_low_1d_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 1d Donchian upper break up OR price > 1d EMA50 (trend flip)
            if close[i] > donch_high_1d_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0