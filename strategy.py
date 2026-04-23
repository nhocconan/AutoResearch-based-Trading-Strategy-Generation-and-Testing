#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
- Long: Close breaks above 20-day high + price > 1w EMA50 (uptrend) + volume > 1.5x 20-day average
- Short: Close breaks below 20-day low + price < 1w EMA50 (downtrend) + volume > 1.5x 20-day average
- Exit: Close retreats below 10-day EMA (for longs) or above 10-day EMA (for shorts)
- Uses Donchian channels for structure, weekly EMA for trend filter, volume for confirmation
- Discrete position sizing (0.25) to minimize fee churn
- Target: 15-30 trades/year (60-120 over 4 years) to avoid fee drag
- Donchian breakouts work in both bull (continuation) and bear (mean reversion via exits) markets
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Calculate 20-day Donchian channels from daily data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian channels: 20-period high and low
    donchian_high = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_1d['close']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1d timeframe (use previous day's levels)
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    
    # Calculate 10-day EMA for exit signal
    ema10_1d = pd.Series(df_1d['close']).ewm(span=10, adjust=False, min_periods=10).mean().values
    ema10_aligned = align_htf_to_ltf(prices, df_1d, ema10_1d)
    
    # Volume confirmation: > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20, 10)  # EMA50 needs 50, Donchian needs 20, EMA10 needs 10
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema10_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1w EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: Close breaks above 20-day high + uptrend + volume spike
        # Short: Close breaks below 20-day low + downtrend + volume spike
        long_signal = (close[i] > donchian_high_aligned[i] and 
                      uptrend and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (close[i] < donchian_low_aligned[i] and 
                       downtrend and
                       volume[i] > 1.5 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Close retreats below 10-day EMA (for longs) or above 10-day EMA (for shorts)
            exit_signal = False
            
            if position == 1:
                # Exit long: Close moves back below 10-day EMA
                if close[i] <= ema10_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Close moves back above 10-day EMA
                if close[i] >= ema10_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirmation"
timeframe = "1d"
leverage = 1.0