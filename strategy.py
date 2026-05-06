#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray + 1d Regime Filter
# Long when Bull Power > 0 AND Bear Power < 0 AND price > 1d EMA34 (bull regime)
# Short when Bull Power < 0 AND Bear Power > 0 AND price < 1d EMA34 (bear regime)
# Elder Ray = Bull Power (high - EMA13), Bear Power (low - EMA13) on 6h timeframe
# Uses discrete sizing 0.25 to manage drawdown in volatile markets
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Elder Ray measures bull/bear strength relative to EMA13; 1d EMA34 filters regime to avoid counter-trend trades

name = "6h_ElderRay_1dEMA34_Regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Get 6h EMA13 for Elder Ray calculation
    close_series = pd.Series(close)
    ema_13 = close_series.ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema_13  # Bull Power: high - EMA13
    bear_power = low - ema_13   # Bear Power: low - EMA13
    
    # Get 1d data ONCE before loop for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for regime filter
    close_1d_series = pd.Series(close_1d)
    ema_34_1d = close_1d_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align HTF indicators to 6h timeframe (wait for completed HTF bar)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Bull Power > 0 (strong buying) AND Bear Power < 0 (weak selling) AND bull regime (price > 1d EMA34)
            if bull_power[i] > 0 and bear_power[i] < 0 and close[i] > ema_34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bull Power < 0 (weak buying) AND Bear Power > 0 (strong selling) AND bear regime (price < 1d EMA34)
            elif bull_power[i] < 0 and bear_power[i] > 0 and close[i] < ema_34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Bull Power <= 0 (buying weakness) OR Bear Power >= 0 (selling strength)
            if bull_power[i] <= 0 or bear_power[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Bull Power >= 0 (buying strength) OR Bear Power <= 0 (selling weakness)
            if bull_power[i] >= 0 or bear_power[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals