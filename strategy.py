#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R Extreme Reversion + 12h EMA50 Trend + Volume Spike
Williams %R identifies overbought/oversold conditions. In strong trends (12h EMA50), 
extreme readings (%R < -90 for longs, %R > -10 for shorts) with volume confirmation 
provide high-probability mean reversion entries. 6h timeframe balances signal quality 
and trade frequency (~15-25 trades/year). Works in both bull/bear markets by 
aligning with higher timeframe trend.
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
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate Williams %R (14 period) on 6h data
    if len(high) < 14:
        return np.zeros(n)
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    # Handle division by zero when high == low
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume confirmation: > 1.8x 24-period average (4 days on 6h)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 24)  # need EMA50_12h, vol MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R deeply oversold (< -90) AND price > 12h EMA50 (uptrend) AND volume spike
            if (williams_r[i] < -90 and 
                close[i] > ema_50_12h_aligned[i] and 
                volume[i] > 1.8 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R deeply overbought (> -10) AND price < 12h EMA50 (downtrend) AND volume spike
            elif (williams_r[i] > -10 and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume[i] > 1.8 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Williams %R returns to neutral range (-50) OR loss of trend alignment
            exit_signal = False
            if position == 1:
                # Exit long when Williams %R > -50 (returns from oversold) OR price < 12h EMA50
                if williams_r[i] > -50 or close[i] < ema_50_12h_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short when Williams %R < -50 (returns from overbought) OR price > 12h EMA50
                if williams_r[i] < -50 or close[i] > ema_50_12h_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Extreme_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0