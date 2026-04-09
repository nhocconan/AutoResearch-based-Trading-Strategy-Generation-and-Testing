#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Supertrend + 1w/1d HTF filter using EMA crossover
# Supertrend(ATR=10, mult=3) on 6h for trend direction and entry timing
# 1w EMA(34) vs 1d EMA(89) for HTF regime: bullish when weekly > daily
# In bullish HTF regime: only take Supertrend longs
# In bearish HTF regime: only take Supertrend shorts
# Position size 0.25 to limit drawdown
# Target: 50-150 total trades over 4 years (12-37/year) to minimize fee drag
# Works in both bull/bear: adapts via HTF EMA crossover filter

name = "6h_1w_1d_supertrend_ema_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Load 1w and 1d data ONCE before loop for EMA
    df_1w = get_htf_data(prices, '1w')
    df_1d = get_htf_data(prices, '1d')
    if len(df_1w) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA(34)
    close_1w = df_1w['close'].values
    ema_34_1w = np.full(len(df_1w), np.nan)
    multiplier_1w = 2 / (34 + 1)
    ema_34_1w[0] = close_1w[0]
    for i in range(1, len(df_1w)):
        ema_34_1w[i] = (close_1w[i] * multiplier_1w) + (ema_34_1w[i-1] * (1 - multiplier_1w))
    
    # Calculate 1d EMA(89)
    close_1d = df_1d['close'].values
    ema_89_1d = np.full(len(df_1d), np.nan)
    multiplier_1d = 2 / (89 + 1)
    ema_89_1d[0] = close_1d[0]
    for i in range(1, len(df_1d)):
        ema_89_1d[i] = (close_1d[i] * multiplier_1d) + (ema_89_1d[i-1] * (1 - multiplier_1d))
    
    # Align HTF EMA values to 6h timeframe
    ema_34_1w_6h = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    ema_89_1d_6h = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    
    # Calculate Supertrend on 6h
    # ATR(10)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr0 = high[i] - low[i]
        tr1 = abs(high[i] - close[i-1])
        tr2 = abs(low[i] - close[i-1])
        tr[i] = max(tr0, tr1, tr2)
    
    atr_10 = np.full(n, np.nan)
    for i in range(9, n):
        if i == 9:
            atr_10[i] = np.mean(tr[:10])
        else:
            atr_10[i] = (atr_10[i-1] * 9 + tr[i]) / 10
    
    # Supertrend calculation
    hl2 = (high + low) / 2
    upper_band = hl2 + (3 * atr_10)
    lower_band = hl2 - (3 * atr_10)
    
    supertrend = np.full(n, np.nan)
    direction = np.full(n, np.nan)  # 1 for uptrend, -1 for downtrend
    
    for i in range(10, n):
        if np.isnan(atr_10[i]) or np.isnan(hl2[i]):
            continue
            
        # Upper band logic
        if i == 10:
            upper_band[i] = hl2[i] + (3 * atr_10[i])
            lower_band[i] = hl2[i] - (3 * atr_10[i])
        else:
            if upper_band[i-1] > close[i-1]:
                upper_band[i] = min(upper_band[i], upper_band[i-1])
            else:
                upper_band[i] = hl2[i] + (3 * atr_10[i])
                
            if lower_band[i-1] < close[i-1]:
                lower_band[i] = max(lower_band[i], lower_band[i-1])
            else:
                lower_band[i] = hl2[i] - (3 * atr_10[i])
        
        # Supertrend logic
        if i == 10:
            if close[i] > upper_band[i]:
                supertrend[i] = lower_band[i]
                direction[i] = -1  # downtrend
            else:
                supertrend[i] = upper_band[i]
                direction[i] = 1   # uptrend
        else:
            if supertrend[i-1] == upper_band[i-1]:
                if close[i] <= upper_band[i]:
                    supertrend[i] = upper_band[i]
                    direction[i] = 1
                else:
                    supertrend[i] = lower_band[i]
                    direction[i] = -1
            else:  # supertrend[i-1] == lower_band[i-1]
                if close[i] >= lower_band[i]:
                    supertrend[i] = lower_band[i]
                    direction[i] = 1
                else:
                    supertrend[i] = upper_band[i]
                    direction[i] = -1
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):  # Start after warmup
        # Skip if any required data is invalid
        if (np.isnan(ema_34_1w_6h[i]) or 
            np.isnan(ema_89_1d_6h[i]) or 
            np.isnan(supertrend[i]) or 
            np.isnan(direction[i])):
            signals[i] = 0.0
            continue
        
        # HTF regime: bullish when weekly EMA > daily EMA
        htf_bullish = ema_34_1w_6h[i] > ema_89_1d_6h[i]
        
        if position == 1:  # Long position
            # Exit conditions
            if htf_bullish:
                # In bullish HTF: exit when Supertrend turns down
                if direction[i] == -1:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = 0.25
            else:
                # In bearish HTF: exit long immediately (shouldn't happen, but safety)
                position = 0
                signals[i] = 0.0
                    
        elif position == -1:  # Short position
            # Exit conditions
            if not htf_bullish:
                # In bearish HTF: exit when Supertrend turns up
                if direction[i] == 1:
                    position = 0
                    signals[i] = 0.0
                else:
                    signals[i] = -0.25
            else:
                # In bullish HTF: exit short immediately (shouldn't happen, but safety)
                position = 0
                signals[i] = 0.0
        else:  # Flat
            # Entry logic based on HTF regime and Supertrend
            if htf_bullish:
                # Bullish HTF: only take longs when Supertrend is up
                if direction[i] == 1:
                    position = 1
                    signals[i] = 0.25
            else:
                # Bearish HTF: only take shorts when Supertrend is down
                if direction[i] == -1:
                    position = -1
                    signals[i] = -0.25
    
    return signals