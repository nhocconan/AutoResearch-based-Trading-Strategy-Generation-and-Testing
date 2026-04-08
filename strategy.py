#!/usr/bin/env python3
# 6h_ema_crossover_12h_volume_momentum
# Hypothesis: EMA(9) crossover on 6h with 12h EMA(50) trend filter and volume momentum filter.
# Long when 6h EMA9 crosses above EMA21, price > 12h EMA50, and volume > 1.3x 6-period average.
# Short when 6h EMA9 crosses below EMA21, price < 12h EMA50, and volume > 1.3x 6-period average.
# Exit on opposite EMA crossover.
# Designed to capture momentum shifts with higher timeframe trend alignment.
# Target: 60-120 total trades over 4 years (~15-30/year).

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ema_crossover_12h_volume_momentum"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate 6h EMAs (9 and 21) for crossover
    ema_9 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema_21 = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean().values
    
    # Calculate average volume for momentum filter (6-period)
    avg_volume = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 21
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(ema_9[i]) or np.isnan(ema_21[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(avg_volume[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: EMA9 crosses below EMA21
            if ema_9[i] < ema_21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: EMA9 crosses above EMA21
            if ema_9[i] > ema_21[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Volume momentum: current volume > 1.3x 6-period average
            volume_momentum = volume[i] > 1.3 * avg_volume[i]
            
            # EMA crossover signals
            ema_cross_up = ema_9[i] > ema_21[i] and ema_9[i-1] <= ema_21[i-1]
            ema_cross_down = ema_9[i] < ema_21[i] and ema_9[i-1] >= ema_21[i-1]
            
            # Entry conditions with 12h trend filter
            if ema_cross_up and (close[i] > ema_50_12h_aligned[i]) and volume_momentum:
                position = 1
                signals[i] = 0.25
            elif ema_cross_down and (close[i] < ema_50_12h_aligned[i]) and volume_momentum:
                position = -1
                signals[i] = -0.25
    
    return signals