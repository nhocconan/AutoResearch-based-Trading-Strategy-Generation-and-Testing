#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation
- Long: Close > Donchian High(20) AND price > 1d EMA34 (uptrend) AND volume > 1.5x 20-period average
- Short: Close < Donchian Low(20) AND price < 1d EMA34 (downtrend) AND volume > 1.5x 20-period average
- Exit: Close < Donchian Low(10) for long OR Close > Donchian High(10) for short
- Uses discrete position sizing (0.25) to minimize fee churn
- Target: 20-50 trades/year (80-200 over 4 years) to avoid fee drag
- Donchian channels provide objective breakout levels; EMA34 filter ensures trend alignment
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
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    # Calculate 1d EMA34 for trend filter
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    # Donchian High(20) = max(high over last 20 periods)
    # Donchian Low(20) = min(low over last 20 periods)
    high_roll_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian High(10) and Low(10) for exit
    high_roll_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_roll_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20)  # EMA34 needs 34, Donchian(20) needs 20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema34_aligned[i]) or 
            np.isnan(high_roll_20[i]) or 
            np.isnan(low_roll_20[i]) or
            np.isnan(high_roll_10[i]) or 
            np.isnan(low_roll_10[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend from 1d EMA34
        uptrend = close[i] > ema34_aligned[i]
        downtrend = close[i] < ema34_aligned[i]
        
        # Donchian breakout signals with trend filter and volume confirmation
        # Long: Close > Donchian High(20) AND uptrend AND volume spike
        # Short: Close < Donchian Low(20) AND downtrend AND volume spike
        long_signal = (close[i] > high_roll_20[i] and 
                      uptrend and
                      volume[i] > 1.5 * vol_ma[i])
        
        short_signal = (close[i] < low_roll_20[i] and 
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
            # Exit conditions: Price retrace to Donchian(10) opposite side
            exit_signal = False
            
            if position == 1:
                # Exit long: Close < Donchian Low(10)
                if close[i] < low_roll_10[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Close > Donchian High(10)
                if close[i] > high_roll_10[i]:
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