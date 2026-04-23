#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
- Long: Close breaks above 20-period Donchian high + price > 1d EMA50 (uptrend) + volume > 1.8x 20-period average
- Short: Close breaks below 20-period Donchian low + price < 1d EMA50 (downtrend) + volume > 1.8x 20-period average
- Exit: Close retouches Donchian midpoint OR trend reversal
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 12-30 trades/year (50-120 over 4 years) to avoid fee drag
- Donchian channels provide clear structure; breakouts with volume work in both bull and bear markets
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
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate 12h Donchian channels (20-period)
    # Donchian High = max(high, lookback=20)
    # Donchian Low = min(low, lookback=20)
    # Donchian Mid = (Donchian High + Donchian Low) / 2
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, Donchian needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: Close breaks above Donchian High + uptrend + volume spike
        # Short: Close breaks below Donchian Low + downtrend + volume spike
        long_signal = (close[i] > donchian_high[i] and 
                      uptrend and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (close[i] < donchian_low[i] and 
                       downtrend and
                       volume[i] > 1.8 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Close retouches Donchian Midpoint OR trend reversal
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

name = "12h_Donchian20_1dEMA50_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0