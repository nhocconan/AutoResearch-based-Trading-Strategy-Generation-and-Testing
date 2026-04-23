#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Uses weekly EMA(50) for trend direction (bullish if price > EMA50, bearish if price < EMA50)
- Donchian(20) breakout on 1d timeframe for entry signals
- Volume spike >1.8x 20-period average to confirm institutional participation
- Position size: 0.25 discrete level to minimize fee churn
- Designed for 7-25 trades/year on 1d timeframe to avoid fee drag
- Works in both bull and bear markets via trend filter (only longs in uptrend, shorts in downtrend)
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
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Donchian(20) channels on 1d data
    donchian_high_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all indicators to 1d timeframe
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_20)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Volume MA, Donchian, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high_20_aligned[i]) or
            np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        # Donchian breakout signals (using current close vs prior channels)
        breakout_up = close[i] > donchian_high_20_aligned[i-1]  # Close above prior 1d Donchian high
        breakout_down = close[i] < donchian_low_20_aligned[i-1]  # Close below prior 1d Donchian low
        
        # Trend filter: EMA50 alignment
        trend_up = close[i] > ema_50_1w_aligned[i]   # Bullish trend
        trend_down = close[i] < ema_50_1w_aligned[i]  # Bearish trend
        
        if position == 0:
            # Long: Donchian breakout up AND volume confirmation AND bullish trend
            if breakout_up and volume_confirm and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Donchian breakout down AND volume confirmation AND bearish trend
            elif breakout_down and volume_confirm and trend_down:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Donchian breakout down (opposite direction)
            if breakout_down:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Donchian breakout up (opposite direction)
            if breakout_up:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_Trend_VolumeConfirmation_v1"
timeframe = "1d"
leverage = 1.0