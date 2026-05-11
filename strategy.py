#!/usr/bin/env python3
"""
1d_1w_Camarilla_R1S1_Breakout_TrendVolume
Hypothesis: Price breaking above weekly R1 or below weekly S1 with daily trend confirmation (EMA100) and volume spike. Uses weekly pivot levels as strong support/resistance. In uptrend (price > EMA100), buy breakouts above R1; in downtrend (price < EMA100), sell breakdowns below S1. Volume confirms institutional interest. Designed for daily timeframe with daily trend filter and weekly pivots to reduce trades and increase win rate. Works in both bull (breakouts) and bear (breakdowns) markets.
"""

name = "1d_1w_Camarilla_R1S1_Breakout_TrendVolume"
timeframe = "1d"
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
    
    # Weekly data for pivot points
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 5:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Using previous week's OHLC
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Shift to use previous week's data (avoid look-ahead)
    w_high_prev = np.roll(w_high, 1)
    w_low_prev = np.roll(w_low, 1)
    w_close_prev = np.roll(w_close, 1)
    # First period: use current values to avoid NaN
    w_high_prev[0] = w_high[0]
    w_low_prev[0] = w_low[0]
    w_close_prev[0] = w_close[0]
    
    # Pivot point calculation
    pivot = (w_high_prev + w_low_prev + w_close_prev) / 3.0
    # R1 and S1 levels
    r1 = pivot + 1 * (w_high_prev - w_low_prev)
    s1 = pivot - 1 * (w_high_prev - w_low_prev)
    
    # Align weekly R1/S1 to daily timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1)
    
    # Daily trend filter (EMA 100)
    ema_100 = pd.Series(close).ewm(
        span=100, adjust=False, min_periods=100
    ).mean().values
    
    # Volume confirmation (20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_100[i]) or np.isnan(vol_ratio[i])):
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
            # Long: break above weekly R1 + above daily EMA100 + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_100[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S1 + below daily EMA100 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_100[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to pivot or trend reversal
            # Calculate weekly pivot for exit (use previous week's data)
            pivot_val = (w_high_prev + w_low_prev + w_close_prev) / 3.0
            pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_val)
            
            if position == 1:
                # Exit long: price returns to weekly pivot OR trend turns down
                if (close[i] <= pivot_aligned[i]) or \
                   (close[i] < ema_100[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to weekly pivot OR trend turns up
                if (close[i] >= pivot_aligned[i]) or \
                   (close[i] > ema_100[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals