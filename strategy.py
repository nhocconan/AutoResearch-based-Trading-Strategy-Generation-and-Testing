#!/usr/bin/env python3
"""
4h_1d_Camarilla_R1S1_Breakout_TrendFilter
Hypothesis: Price breaking above daily R1 or below daily S1 with daily trend confirmation (EMA50) and volume spike. Uses daily Camarilla levels as strong support/resistance. In uptrend (price > EMA50), buy breakouts above R1; in downtrend (price < EMA50), sell breakdowns below S1. Volume confirms institutional interest. Designed for 4h timeframe with daily trend filter and daily pivots to reduce trades and increase win rate. Works in both bull (breakouts) and bear (breakdowns) markets.
"""

name = "4h_1d_Camarilla_R1S1_Breakout_TrendFilter"
timeframe = "4h"
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
    
    # Calculate daily Camarilla pivot points using previous period's OHLC
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
    r1 = d_close_prev + (1.1/12) * (d_high_prev - d_low_prev)
    s1 = d_close_prev - (1.1/12) * (d_high_prev - d_low_prev)
    
    # Align daily R1/S1 to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    
    # Daily trend filter (EMA 50)
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation (20-period average on 4h)
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
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_spike = vol_ratio[i] > 2.0
        
        if position == 0:
            # Long: break above daily R1 + above daily EMA50 + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below daily S1 + below daily EMA50 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to daily pivot or trend reversal
            if position == 1:
                # Exit long: price returns to daily pivot OR trend turns down
                if (close[i] <= pivot_aligned[i]) or \
                   (close[i] < ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to daily pivot OR trend turns up
                if (close[i] >= pivot_aligned[i]) or \
                   (close[i] > ema_50_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals