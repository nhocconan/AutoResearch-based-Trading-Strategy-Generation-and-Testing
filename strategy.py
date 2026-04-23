#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA50 trend filter and volume spike confirmation.
- Uses Donchian channel (20-period high/low) from 4h for breakout signals
- 1d EMA50 as trend filter (long only above, short only below)
- Volume > 2.0x 20-period average for confirmation
- Position size: 0.25 discrete level to minimize fee churn
- Target: 20-50 trades/year on 4h timeframe (80-200 total over 4 years)
- Works in both bull/bear via trend filter + volatility-adjusted breakouts
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
    
    # 4h data for Donchian channel calculation (20-period)
    high_4h = get_htf_data(prices, '4h')['high'].values
    low_4h = get_htf_data(prices, '4h')['low'].values
    
    # Calculate Donchian channel (20-period high/low) from 4h
    # Donchian High = max(high over last 20 periods)
    # Donchian Low = min(low over last 20 periods)
    donchian_high = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (using completed 4h bar)
    donchian_high_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, get_htf_data(prices, '4h'), donchian_low)
    
    # 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Donchian(20), EMA(50)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high_aligned[i]  # Close above upper band
        breakout_down = close[i] < donchian_low_aligned[i]  # Close below lower band
        
        if position == 0:
            # Long: 4h Donchian breakout up AND price above 1d EMA50 AND volume confirmation
            if breakout_up and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 4h Donchian breakout down AND price below 1d EMA50 AND volume confirmation
            elif breakout_down and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 4h Donchian breakdown OR price crosses below 1d EMA50
            if breakout_down or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 4h Donchian breakout OR price crosses above 1d EMA50
            if breakout_up or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_VolumeSpike_Filter_v1"
timeframe = "4h"
leverage = 1.0