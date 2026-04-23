#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
- Long: Close breaks above Donchian upper band (20-period high) + price > 12h EMA50 + volume > 1.8x 20-period average
- Short: Close breaks below Donchian lower band (20-period low) + price < 12h EMA50 + volume > 1.8x 20-period average
- Exit: Close retreats below midpoint of Donchian channel
- Uses Donchian channels for structure, 12h EMA for trend filter, volume for confirmation
- Discrete position sizing (0.25) to minimize fee churn
- Target: 30-60 trades/year (120-240 over 4 years) to avoid fee drag
- Works in both bull and bear markets: trend filter prevents counter-trend trades, volume confirms legitimacy
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 for trend filter
    ema50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    donchian_mid = (donchian_upper + donchian_lower) / 2.0
    
    # Volume confirmation: > 1.8x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # EMA50 needs 50, Donchian needs 20, volume MA needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or 
            np.isnan(donchian_upper[i]) or 
            np.isnan(donchian_lower[i]) or 
            np.isnan(donchian_mid[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 12h EMA50
        uptrend = close[i] > ema50_aligned[i]
        downtrend = close[i] < ema50_aligned[i]
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: Close breaks above upper band + uptrend + volume spike
        # Short: Close breaks below lower band + downtrend + volume spike
        long_signal = (close[i] > donchian_upper[i] and 
                      uptrend and
                      volume[i] > 1.8 * vol_ma[i])
        
        short_signal = (close[i] < donchian_lower[i] and 
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
            # Exit conditions: Close retreats below midpoint of Donchian channel
            exit_signal = False
            
            if position == 1:
                # Exit long: Close moves back below midpoint
                if close[i] <= donchian_mid[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Close moves back above midpoint
                if close[i] >= donchian_mid[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50_Trend_VolumeConfirmation"
timeframe = "4h"
leverage = 1.0