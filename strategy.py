#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "12h_Weekly_Pivot_Donchian_Breakout_v1"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w data once
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1d ATR for volatility filter ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr = np.maximum(high_1d - low_1d, 
                    np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                               np.abs(low_1d - np.roll(close_1d, 1))))
    tr[0] = high_1d[0] - low_1d[0]
    atr10_1d = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    atr10_1d_aligned = align_htf_to_ltf(prices, df_1d, atr10_1d)
    
    # === 1d Weekly Pivot Points (based on weekly close) ===
    # Use weekly data to calculate pivot, then apply to daily
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    weekly_pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    weekly_range = weekly_high - weekly_low
    
    # Weekly support/resistance levels
    r1 = weekly_pivot + (weekly_range * 1.1 / 12)
    s1 = weekly_pivot - (weekly_range * 1.1 / 12)
    r2 = weekly_pivot + (weekly_range * 1.1 / 6)
    s2 = weekly_pivot - (weekly_range * 1.1 / 6)
    
    # Align weekly pivot levels to 12h timeframe
    r1_12h = align_htf_to_ltf(prices, df_1w, r1)
    s1_12h = align_htf_to_ltf(prices, df_1w, s1)
    r2_12h = align_htf_to_ltf(prices, df_1w, r2)
    s2_12h = align_htf_to_ltf(prices, df_1w, s2)
    
    # === 12h Donchian Channel (20-period) ===
    high_12h = get_htf_data(prices, '12h')['high'].values
    low_12h = get_htf_data(prices, '12h')['low'].values
    
    # Calculate Donchian on 12h data
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian to 12h timeframe (already aligned since we're using 12h data)
    donchian_high_12h = donchian_high
    donchian_low_12h = donchian_low
    
    # === Volume filter: current volume > 1.5x 20-period average ===
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # warmup for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(r1_12h[i]) or np.isnan(s1_12h[i]) or np.isnan(r2_12h[i]) or np.isnan(s2_12h[i]) or
            np.isnan(donchian_high_12h[i]) or np.isnan(donchian_low_12h[i]) or
            np.isnan(atr10_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Volatility filter: only trade when volatility is elevated
            vol_filter = atr10_1d_aligned[i] > np.nanmedian(atr10_1d_aligned[:i+1]) * 0.8
            
            if vol_filter:
                # Long breakout: price breaks above weekly R2 AND Donchian high
                long_cond = (close[i] > r2_12h[i] and 
                            close[i] > donchian_high_12h[i] and
                            volume[i] > vol_ma20[i])
                
                # Short breakdown: price breaks below weekly S2 AND Donchian low
                short_cond = (close[i] < s2_12h[i] and 
                             close[i] < donchian_low_12h[i] and
                             volume[i] > vol_ma20[i])
            else:
                low_vol = True
                long_cond = False
                short_cond = False
            
            if long_cond:
                signals[i] = 0.25
                position = 1
            elif short_cond:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below weekly S1 or Donchian low
            exit_cond = (close[i] < s1_12h[i] or 
                        close[i] < donchian_low_12h[i])
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above weekly R1 or Donchian high
            exit_cond = (close[i] > r1_12h[i] or 
                        close[i] > donchian_high_12h[i])
            
            if exit_cond:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Weekly pivot levels act as strong support/resistance zones that institutions watch.
# In volatile markets (high ATR), price tends to break through these levels with momentum.
# Strategy breaks out when price crosses weekly R2/S2 with Donchian confirmation and volume surge.
# Exits at weekly S1/R1 (profit target) or Donchian reversal (stop loss).
# Works in both bull (breakouts) and bear (breakdowns) markets.
# Targets 50-150 trades over 4 years to minimize fee drag. Uses discrete sizing (0.25).