#!/usr/bin/env python3
# 1d_1w_camarilla_reversion_v1
# Strategy: Daily Camarilla pivot mean reversion with weekly trend filter and volume confirmation
# Timeframe: 1d
# Leverage: 1.0
# Hypothesis: In ranging markets, price reverts to Camarilla pivot levels (H3/L3). 
# In trending markets, weekly EMA20 filters direction. Volume confirms institutional interest.
# Low frequency (~10-25/year) to minimize fee drag. Works in both bull and bear regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1d_1w_camarilla_reversion_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Weekly EMA(20) for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(30, n):
        # Need 24h of data for Camarilla calculation
        if i < 1:
            signals[i] = 0.0
            continue
            
        # Previous day's OHLC for Camarilla levels
        ph = high[i-1]
        pl = low[i-1]
        pc = close[i-1]
        
        # Calculate Camarilla levels
        range_val = ph - pl
        if range_val <= 0:
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
            
        # Resistance levels
        h3 = pc + (range_val * 1.1 / 4)
        h4 = pc + (range_val * 1.1 / 2)
        # Support levels
        l3 = pc - (range_val * 1.1 / 4)
        l4 = pc - (range_val * 1.1 / 2)
        
        # Weekly trend filter
        weekly_uptrend = close[i] > ema_20_1w_aligned[i]
        weekly_downtrend = close[i] < ema_20_1w_aligned[i]
        
        # Volume confirmation: current volume > 1.5x 20-day average
        if i >= 20:
            vol_avg_20 = np.mean(volume[i-20:i])
            vol_confirm = volume[i] > (1.5 * vol_avg_20)
        else:
            vol_confirm = False
        
        # Entry logic: Mean reversion at extreme levels with trend alignment
        if (close[i] <= l3 and weekly_uptrend and vol_confirm and position != 1):
            # Long at L3 support in uptrend with volume
            position = 1
            signals[i] = 0.25
        elif (close[i] >= h3 and weekly_downtrend and vol_confirm and position != -1):
            # Short at H3 resistance in downtrend with volume
            position = -1
            signals[i] = -0.25
        # Exit: Return to mean (pivot) or trend reversal
        elif position == 1 and (close[i] >= pc or not weekly_uptrend):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= pc or not weekly_downtrend):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals