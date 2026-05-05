#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + Weekly Regime Filter
# Long when Bull Power > 0 AND Bear Power < 0 AND weekly close > weekly EMA34 (bull regime)
# Short when Bear Power < 0 AND Bull Power < 0 AND weekly close < weekly EMA34 (bear regime)
# Exit when Elder Ray signals weaken (Bull Power <= 0 for long, Bear Power >= 0 for short)
# Uses 6h primary timeframe with 1w HTF for regime filter
# Elder Ray measures bull/bear power relative to EMA13; regime filter ensures trades align with weekly trend
# Discrete sizing (0.25) to limit fee drag and manage drawdown
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe

name = "6h_ElderRay_WeeklyRegime"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Get 1w data ONCE before loop for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 40:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for regime filter
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    weekly_bull_regime = close_1w > ema_34_1w
    weekly_bear_regime = close_1w < ema_34_1w
    
    # Align weekly regime to 6h timeframe
    weekly_bull_regime_aligned = align_htf_to_ltf(prices, df_1w, weekly_bull_regime)
    weekly_bear_regime_aligned = align_htf_to_ltf(prices, df_1w, weekly_bear_regime)
    
    # Calculate 6h EMA13 for Elder Ray
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or 
            np.isnan(weekly_bull_regime_aligned[i]) or 
            np.isnan(weekly_bear_regime_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: Bull Power > 0 AND Bear Power < 0 AND weekly bull regime
            if (bull_power[i] > 0 and 
                bear_power[i] < 0 and 
                weekly_bull_regime_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: Bear Power < 0 AND Bull Power < 0 AND weekly bear regime
            elif (bear_power[i] < 0 and 
                  bull_power[i] < 0 and 
                  weekly_bear_regime_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 (weakening bullish momentum)
            if bull_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bear Power >= 0 (weakening bearish momentum)
            if bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals