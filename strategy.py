#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Strategy: 6h Williams %R + 12h EMA Trend Filter with Volume Spike
# Hypothesis: Williams %R identifies overbought/oversold reversals; 12h EMA filters trend direction; volume spike confirms momentum.
# Works in bull via trend-aligned reversals, in bear via mean-reversion at extremes with volume confirmation.
# Target: 50-150 total trades over 4 years (~12-37/year) to minimize fee drag.
name = "6h_williamsr_12h_ema_volume_v1"
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
    
    # Get 12h data for EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 21:
        return np.zeros(n)
    
    # Calculate 12h EMA(21) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=21, adjust=False).mean().values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Calculate Williams %R (14-period) on 6h
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)
    # Handle division by zero (when high == low)
    williams_r = np.where((highest_high - lowest_low) == 0, -50, williams_r)
    
    # Volume spike: current volume > 1.5 x 20-period volume average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # Track position: 1=long, -1=short, 0=flat
    
    for i in range(14, n):
        # Skip if required data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(williams_r[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: Williams %R exits oversold OR volume spike fades OR trend turns bearish
            if williams_r[i] > -20 or not vol_spike[i] or close[i] < ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Maintain long position
        elif position == -1:  # Short position
            # Exit: Williams %R exits overbought OR volume spike fades OR trend turns bullish
            if williams_r[i] < -80 or not vol_spike[i] or close[i] > ema_12h_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Maintain short position
        else:  # Flat, look for entry
            # Enter long: Williams %R oversold (< -80) + volume spike + bullish trend (price > 12h EMA)
            if williams_r[i] < -80 and vol_spike[i] and close[i] > ema_12h_aligned[i]:
                position = 1
                signals[i] = 0.25
            # Enter short: Williams %R overbought (> -20) + volume spike + bearish trend (price < 12h EMA)
            elif williams_r[i] > -20 and vol_spike[i] and close[i] < ema_12h_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals