#!/usr/bin/env python3
"""
Hypothesis: 6h Elder Ray Bull/Bear Power with 1d Williams %R regime filter and volume confirmation.
Long when Bull Power > 0 AND Bear Power < 0 (bullish momentum) AND daily Williams %R < -80 (oversold) AND volume > 1.5x average.
Short when Bear Power < 0 AND Bull Power > 0 (bearish momentum) AND daily Williams %R > -20 (overbought) AND volume > 1.5x average.
Exit when Elder Ray power diverges (Bull Power <= 0 for long, Bear Power >= 0 for short).
Uses discrete position sizing (0.25) to minimize fee churn. Targets 20-40 trades/year per symbol.
Elder Ray measures bull/bear power via EMA13, Williams %R identifies overextended conditions for mean reversion entry.
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
    
    # Load 1d data for Williams %R regime filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate Williams %R on 1d data (14-period)
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r_1d = -100 * (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14)
    # Handle division by zero
    williams_r_1d = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r_1d)
    
    # Align Williams %R to 6h timeframe
    williams_r_1d_aligned = align_htf_to_ltf(prices, df_1d, williams_r_1d)
    
    # Calculate Elder Ray on 6h data: Bull Power = High - EMA13, Bear Power = Low - EMA13
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume average (20-period) on primary timeframe
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(williams_r_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        vol_current = volume[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND Bear Power < 0 (bullish) AND Williams %R < -80 (oversold) AND volume confirmation
            if (bull_power[i] > 0 and bear_power[i] < 0 and 
                williams_r_1d_aligned[i] < -80 and 
                vol_current > 1.5 * vol_ma_val):
                signals[i] = 0.25
                position = 1
            # Short: Bear Power < 0 AND Bull Power > 0 (bearish) AND Williams %R > -20 (overbought) AND volume confirmation
            elif (bear_power[i] < 0 and bull_power[i] > 0 and 
                  williams_r_1d_aligned[i] > -20 and 
                  vol_current > 1.5 * vol_ma_val):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: Elder Ray power diverges
            exit_signal = False
            
            if position == 1:
                # Exit long: Bull Power <= 0 (bullish momentum faded)
                if bull_power[i] <= 0:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Bear Power >= 0 (bearish momentum faded)
                if bear_power[i] >= 0:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_ElderRay_1dWilliamsR_Volume"
timeframe = "6h"
leverage = 1.0