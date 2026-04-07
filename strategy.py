#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray Power with 1d regime filter
# Elder Ray: Bull Power = High - EMA(13), Bear Power = EMA(13) - Low
# In bull regime (1d close > 50 EMA): enter long when Bull Power crosses above 0
# In bear regime (1d close < 50 EMA): enter short when Bear Power crosses above 0
# Exit when power crosses back below 0
# Target: 12-37 trades/year, works in both bull and bear via regime adaptation

name = "6h_elder_ray_1d_regime_v1"
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
    
    # Get 1d data for regime filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 13-period EMA for Elder Ray
    close_s = pd.Series(close)
    ema13 = close_s.ewm(span=13, adjust=False).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # High - EMA13
    bear_power = ema13 - low   # EMA13 - Low
    
    # 1d regime: close > 50 EMA = bull regime, close < 50 EMA = bear regime
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False).mean().values
    bull_regime = close_1d > ema50_1d  # True for bull regime
    bull_regime_aligned = align_htf_to_ltf(prices, df_1d, bull_regime)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(13, n):
        # Skip if required data not available
        if (np.isnan(ema13[i]) or np.isnan(bull_power[i]) or 
            np.isnan(bear_power[i]) or np.isnan(bull_regime_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Bull Power crosses below 0
            if bull_power[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Bear Power crosses below 0
            if bear_power[i] <= 0:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: Bull Power crosses above 0 in bull regime
            if bull_power[i] > 0 and bull_power[i-1] <= 0 and bull_regime_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: Bear Power crosses above 0 in bear regime
            elif bear_power[i] > 0 and bear_power[i-1] <= 0 and not bull_regime_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals