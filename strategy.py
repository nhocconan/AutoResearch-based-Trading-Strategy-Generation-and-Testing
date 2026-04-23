#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation
- Long: Close > Donchian High(20) AND price > 1w EMA50 AND volume > 1.5x 20-period average
- Short: Close < Donchian Low(20) AND price < 1w EMA50 AND volume > 1.5x 20-period average
- Exit: Opposite Donchian breakout (Close < Donchian High(10) for long exit, Close > Donchian Low(10) for short exit)
- Uses 1w EMA50 for trend alignment (avoids counter-trend whipsaws)
- Volume confirmation ensures institutional participation
- Donchian channels provide clear breakout levels with built-in volatility adjustment
- Works in both bull and bear markets by trading with the higher timeframe trend
- Target: 20-50 trades/year (80-200 total over 4 years) to minimize fee drag
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
    
    # Calculate Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20)
    low_roll = pd.Series(low).rolling(window=20, min_periods=20)
    donchian_high = high_roll.max().values
    donchian_low = low_roll.min().values
    
    # Calculate Donchian channels (10-period) for exits
    high_roll_exit = pd.Series(high).rolling(window=10, min_periods=10)
    low_roll_exit = pd.Series(low).rolling(window=10, min_periods=10)
    donchian_high_exit = high_roll_exit.max().values
    donchian_low_exit = low_roll_exit.min().values
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF EMA50 to LTF
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(20, 50)  # Donchian needs 20, EMA50 needs 50
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or 
            np.isnan(donchian_high_exit[i]) or 
            np.isnan(donchian_low_exit[i]) or 
            np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i]
        breakout_down = close[i] < donchian_low[i]
        
        # Trend filter
        uptrend = close[i] > ema50_1w_aligned[i]
        downtrend = close[i] < ema50_1w_aligned[i]
        
        # Volume confirmation
        volume_ok = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Breakout up + uptrend + volume confirmation
            if breakout_up and uptrend and volume_ok:
                signals[i] = 0.25
                position = 1
            # Short: Breakout down + downtrend + volume confirmation
            elif breakout_down and downtrend and volume_ok:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Opposite Donchian breakout (using 10-period for smoother exit)
            exit_signal = False
            
            if position == 1:
                # Exit long: Close < Donchian Low(10)
                if close[i] < donchian_low_exit[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: Close > Donchian High(10)
                if close[i] > donchian_high_exit[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50_Trend_VolumeConfirm"
timeframe = "1d"
leverage = 1.0