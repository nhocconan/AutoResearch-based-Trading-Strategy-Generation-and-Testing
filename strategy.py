#!/usr/bin/env python3
"""
1h_4h1d_Camarilla_R1S1_Breakout_TrendFilter
Hypothesis: Price breaking above daily R1 or below daily S1 with 4h trend confirmation (EMA50) and volume spike. Uses daily Camarilla levels as strong support/resistance. In uptrend (price > EMA50), buy breakouts above R1; in downtrend (price < EMA50), sell breakdowns below S1. Volume confirms institutional interest. Designed for 1h timeframe with 4h trend filter and daily pivots to reduce trades and increase win rate. Works in both bull (breakouts) and bear (breakdowns) markets.
"""

name = "1h_4h1d_Camarilla_R1S1_Breakout_TrendFilter"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price and volume
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Daily data for Camarilla pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 5:
        return np.zeros(n)
    
    # Calculate daily Camarilla pivot points
    # Using previous period's OHLC
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Shift to use previous period's data (avoid look-ahead)
    d_high_prev = np.roll(d_high, 1)
    d_low_prev = np.roll(d_low, 1)
    d_close_prev = np.roll(d_close, 1)
    # First period: use current values to avoid NaN
    d_high_prev[0] = d_high[0]
    d_low_prev[0] = d_low[0]
    d_close_prev[0] = d_close[0]
    
    # Calculate pivot point
    pivot = (d_high_prev + d_low_prev + d_close_prev) / 3.0
    # Calculate Camarilla R1 and S1 levels
    # R1 = Close + 1.1/12 * (High - Low)
    # S1 = Close - 1.1/12 * (High - Low)
    r1 = d_close_prev + (1.1/12) * (d_high_prev - d_low_prev)
    s1 = d_close_prev - (1.1/12) * (d_high_prev - d_low_prev)
    
    # Align daily R1/S1 to 1h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # 4h trend filter (EMA 50)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 5:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation (20-period average on 1h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.20
            elif position == -1:
                signals[i] = -0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: break above daily R1 + above 4h EMA50 + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike):
                signals[i] = 0.20
                position = 1
            # Short: break below daily S1 + below 4h EMA50 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike):
                signals[i] = -0.20
                position = -1
        else:
            # Exit conditions: return to pivot or trend reversal
            # Calculate pivot for exit (use previous period's data)
            pivot_val = (d_high_prev + d_low_prev + d_close_prev) / 3.0
            pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot_val)
            
            if position == 1:
                # Exit long: price returns to daily pivot OR trend turns down
                if (close[i] <= pivot_aligned[i]) or \
                   (close[i] < ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Exit short: price returns to daily pivot OR trend turns up
                if (close[i] >= pivot_aligned[i]) or \
                   (close[i] > ema_50_4h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
    
    return signals