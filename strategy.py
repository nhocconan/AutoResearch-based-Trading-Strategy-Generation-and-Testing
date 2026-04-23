#!/usr/bin/env python3
"""
Hypothesis: 4h Williams %R extreme reversal with 1d EMA trend filter and volume spike
- Williams %R(14) < -80 for oversold long, > -20 for overbought short
- 1d EMA(50) as trend filter: long only when price > EMA, short only when price < EMA
- Volume confirmation (> 1.5x 20-period average) ensures momentum behind reversal
- Designed for 4h timeframe targeting 20-50 trades/year (80-200 over 4 years)
- Works in bull markets by buying oversold dips in uptrend
- Works in bear markets by selling overbought rallies in downtrend
- Williams %R captures exhaustion moves that often reverse sharply
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
    
    # Calculate 4h Williams %R (14-period)
    period = 14
    highest_high = pd.Series(high).rolling(window=period, min_periods=period).max().values
    lowest_low = pd.Series(low).rolling(window=period, min_periods=period).min().values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    denominator = highest_high - lowest_low
    williams_r = np.where(denominator != 0, ((highest_high - close) / denominator) * -100, np.nan)
    
    # Calculate 1d EMA(50) trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(period, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(williams_r[i]) or np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R conditions
        oversold = williams_r[i] < -80
        overbought = williams_r[i] > -20
        
        # Trend filter
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Volume spike
        volume_spike = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: oversold + uptrend + volume spike
            if oversold and uptrend and volume_spike:
                signals[i] = 0.25
                position = 1
            # Short: overbought + downtrend + volume spike
            elif overbought and downtrend and volume_spike:
                signals[i] = -0.25
                position = -1
        else:
            # Exit: opposite Williams %R extreme or trend change
            exit_signal = False
            
            if position == 1:
                # Exit long: overbought or trend turns down
                if overbought or not uptrend:
                    exit_signal = True
            elif position == -1:
                # Exit short: oversold or trend turns up
                if oversold or not downtrend:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsR_Extreme_1dEMA50_Trend_VolumeSpike"
timeframe = "4h"
leverage = 1.0