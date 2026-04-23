#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Uses Donchian channel breakout (20-day high/low) for trend capture
- 1w EMA50 as trend filter (long only above, short only below) - avoids counter-trend whipsaw
- Volume > 1.5x 20-period average for confirmation (reduces false breakouts)
- Position size: 0.25 discrete level to minimize fee churn
- Target: 7-25 trades/year on 1d timeframe (30-100 total over 4 years)
- Works in both bull/bear via trend filter + volatility-adjusted breakouts
- Uses 1w HTF as specified in experiment parameters
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
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 1w data for Donchian channel calculation (HTF as specified)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channel (20-period) from prior 1w bar
    # Upper band = highest high over 20 periods
    # Lower band = lowest low over 20 periods
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1w timeframe (using completed 1w bar)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # 1w data for EMA50 trend filter
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Donchian, EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(donchian_high_aligned[i]) or
            np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        # Donchian breakout signals
        breakout_up = close[i] > donchian_high_aligned[i]  # Close above upper band
        breakout_down = close[i] < donchian_low_aligned[i]  # Close below lower band
        
        if position == 0:
            # Long: 1w Donchian upper breakout up AND price above 1w EMA50 AND volume confirmation
            if breakout_up and close[i] > ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: 1w Donchian lower breakout down AND price below 1w EMA50 AND volume confirmation
            elif breakout_down and close[i] < ema_50_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1w Donchian lower breakdown OR price crosses below 1w EMA50
            if breakout_down or close[i] < ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: 1w Donchian upper breakout OR price crosses above 1w EMA50
            if breakout_up or close[i] > ema_50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_Breakout_1wEMA50_VolumeConfirmation_v1"
timeframe = "1d"
leverage = 1.0