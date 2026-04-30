#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Williams %R regime filter
# Long when Bull Power > 0 and Bear Power < 0 with 1d Williams %R oversold (< -80)
# Short when Bear Power < 0 and Bull Power > 0 with 1d Williams %R overbought (> -20)
# Exit when Elder Ray power crosses zero (mean reversion)
# Uses Elder Ray to measure bull/bear strength relative to EMA13, Williams %R for regime
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_ElderRay_1dWilliamsR_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Load 1d data ONCE before loop for Williams %R regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate 1d Williams %R for regime filter (14-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    williams_r_1d = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r_1d)  # avoid div by zero
    
    # Align 1d Williams %R to 6h timeframe (completed 1d bar only)
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Calculate Elder Ray on 6h timeframe
    # Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # warmup for EMA13 and Williams %R
    
    for i in range(start_idx, n):
        # Skip if indicators not available
        if (np.isnan(ema_13[i]) or 
            np.isnan(williams_r_1d_aligned[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i])):
            signals[i] = 0.0
            continue
        
        curr_bull_power = bull_power[i]
        curr_bear_power = bear_power[i]
        curr_williams_r = williams_r_1d_aligned[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: Bull Power > 0 (strong buying), Williams %R oversold (< -80)
            if (curr_bull_power > 0 and 
                curr_williams_r < -80):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 (strong selling), Williams %R overbought (> -20)
            elif (curr_bear_power < 0 and 
                  curr_williams_r > -20):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position
            # Exit condition: Bull Power crosses below zero (weakening buying pressure)
            if curr_bull_power <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit condition: Bear Power crosses above zero (weakening selling pressure)
            if curr_bear_power >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals