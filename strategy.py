#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Elder Ray (Bull/Bear Power) + 1d EMA200 trend filter + volume confirmation
# Elder Ray measures bull/bear power relative to EMA13: Bull Power = High - EMA13, Bear Power = Low - EMA13
# In bull markets (price > 1d EMA200): look for Bull Power expansion with volume confirmation to go long
# In bear markets (price < 1d EMA200): look for Bear Power expansion with volume confirmation to go short
# Works in both regimes: adapts to trend direction via higher timeframe filter
# Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing 0.25

name = "6h_1d_elder_ray_volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA200 for trend filter
    close_1d = df_1d['close'].values
    ema200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 1d EMA200 to 6h timeframe (wait for 1d bar close)
    ema200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # Calculate 6h EMA13 for Elder Ray
    ema13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components
    bull_power = high - ema13  # Bull Power: High - EMA13
    bear_power = low - ema13   # Bear Power: Low - EMA13
    
    # Volume confirmation: 6h volume > 1.5x 20-period average
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(100, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema200_1d_aligned[i]) or np.isnan(ema13[i]) or
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or
            np.isnan(volume_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        volume_confirmed = volume[i] > 1.5 * volume_ma[i]
        
        # Trend filter from 1d EMA200
        bull_trend = close[i] > ema200_1d_aligned[i]
        bear_trend = close[i] < ema200_1d_aligned[i]
        
        if position == 1:  # Long position
            # Exit: Bull Power turns negative OR trend turns bearish
            if bull_power[i] < 0 or bear_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: Bear Power turns positive OR trend turns bullish
            if bear_power[i] > 0 or bull_trend:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat
            # Entry logic based on trend and Elder Ray power
            if bull_trend and volume_confirmed:
                # In bull trend: go long on Bull Power expansion (making new high)
                if i > 0 and bull_power[i] > bull_power[i-1]:
                    position = 1
                    signals[i] = 0.25
            elif bear_trend and volume_confirmed:
                # In bear trend: go short on Bear Power expansion (making new low)
                if i > 0 and bear_power[i] < bear_power[i-1]:
                    position = -1
                    signals[i] = -0.25
    
    return signals