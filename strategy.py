#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout + 1d EMA34 trend + volume confirmation
- Long: Close breaks above Donchian upper (20-period high) + price > 1d EMA34 (uptrend) + volume > 1.5x 20-period average
- Short: Close breaks below Donchian lower (20-period low) + price < 1d EMA34 (downtrend) + volume > 1.5x 20-period average
- Exit: Close retouches Donchian midpoint OR trend reversal
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag
- Donchian channels provide clear structure; breakouts with volume and trend filter work in both bull and bear markets
- Using 1d EMA34 as HTF trend filter for better alignment with 4h timeframe
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian channels from 4h data (20-period)
    # Upper = 20-period high, Lower = 20-period low, Mid = (Upper + Lower)/2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_upper = high_series.rolling(window=20, min_periods=20).max().values
    donchian_lower = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, Donchian needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA34
        uptrend = close[i] > ema34_aligned[i]
        downtrend = close[i] < ema34_aligned[i]
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: Close breaks above upper + uptrend + volume spike
        # Short: Close breaks below lower + downtrend + volume spike
        long_signal = (close[i] > donchian_upper[i] and 
                      uptrend and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (close[i] < donchian_lower[i] and 
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
            # Exit conditions: Close retouches Donchian midpoint OR trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: Close retouches midpoint or trend turns down
                if (close[i] <= donchian_mid[i] or 
                    not uptrend):
                    exit_signal = True
            elif position == -1:
                # Exit short: Close retouches midpoint or trend turns up
                if (close[i] >= donchian_mid[i] or 
                    not downtrend):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA34_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0