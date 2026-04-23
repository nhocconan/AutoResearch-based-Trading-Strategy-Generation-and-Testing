#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1w EMA(34) trend filter and volume confirmation
- Donchian(20) breakout captures momentum with clear entry/exit levels
- 1w EMA(34) ensures alignment with weekly trend to reduce counter-trend trades
- Volume spike (>2.0x 24-period average) confirms strong participation
- Designed for 12h timeframe targeting 12-37 trades/year (50-150 over 4 years) to minimize fee drag
- Works in both bull and bear markets by trading with the weekly trend when breakout occurs
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
    
    # Get 1w data for EMA trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA(34) for trend filter
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate Donchian(20) channels (primary timeframe)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 2.0x 24-period average (12h * 24 = 12 days)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, 20, 24)  # EMA1w, Donchian, volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1w_aligned[i]) or 
            np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Donchian breakout signals with trend filter
        # Long: price breaks above upper band + uptrend + volume spike
        # Short: price breaks below lower band + downtrend + volume spike
        long_signal = (close[i] > highest_high[i] and 
                      close[i] > ema_34_1w_aligned[i] and
                      volume[i] > 2.0 * vol_ma[i])
        
        short_signal = (close[i] < lowest_low[i] and 
                       close[i] < ema_34_1w_aligned[i] and
                       volume[i] > 2.0 * vol_ma[i])
        
        if position == 0:
            if long_signal:
                signals[i] = 0.25
                position = 1
            elif short_signal:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price crosses middle of Donchian channel or trend reversal
            exit_signal = False
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2
            
            if position == 1:
                # Exit long: price crosses below mid-channel or trend reversal
                if (close[i] < donchian_mid or 
                    close[i] < ema_34_1w_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above mid-channel or trend reversal
                if (close[i] > donchian_mid or 
                    close[i] > ema_34_1w_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12h_Donchian20_1wEMA34_Trend_VolumeConfirm"
timeframe = "12h"
leverage = 1.0