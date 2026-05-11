#!/usr/bin/env python3
"""
4h_12h_Camarilla_Pivot_Breakout_TrendVolume_v2
Hypothesis: Price breaking above monthly R1 or below monthly S1 with 12h trend confirmation (EMA50) and volume spike. Uses monthly pivot levels as strong support/resistance. In uptrend (price > EMA50), buy breakouts above R1; in downtrend (price < EMA50), sell breakdowns below S1. Volume confirms institutional interest. Designed for 4h timeframe with 12h trend filter and monthly pivots to reduce trades and increase win rate. Works in both bull (breakouts) and bear (breakdowns) markets.
"""

name = "4h_12h_Camarilla_Pivot_Breakout_TrendVolume_v2"
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
    
    # Monthly data for pivot points
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate monthly pivot points (standard formula)
    # Using previous period's OHLC
    d_high = df_12h['high'].values
    d_low = df_12h['low'].values
    d_close = df_12h['close'].values
    
    # Shift to use previous period's data (avoid look-ahead)
    d_high_prev = np.roll(d_high, 1)
    d_low_prev = np.roll(d_low, 1)
    d_close_prev = np.roll(d_close, 1)
    # First period: use current values to avoid NaN
    d_high_prev[0] = d_high[0]
    d_low_prev[0] = d_low[0]
    d_close_prev[0] = d_close[0]
    
    # Pivot point calculation
    pivot = (d_high_prev + d_low_prev + d_close_prev) / 3.0
    # R1 and S1 levels
    r1 = pivot + 1 * (d_high_prev - d_low_prev)
    s1 = pivot - 1 * (d_high_prev - d_low_prev)
    
    # Align monthly R1/S1 to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_12h, r1)
    s1_aligned = align_htf_to_ltf(prices, df_12h, s1)
    
    # 12h trend filter (EMA 50)
    ema_50_12h = pd.Series(d_close).ewm(
        span=50, adjust=False, min_periods=50
    ).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
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
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ratio[i])):
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
            # Long: break above monthly R1 + above 12h EMA50 + volume spike
            if (close[i] > r1_aligned[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below monthly S1 + below 12h EMA50 + volume spike
            elif (close[i] < s1_aligned[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to pivot or trend reversal
            # Calculate pivot for exit (use previous period's data)
            pivot_val = (d_high_prev + d_low_prev + d_close_prev) / 3.0
            pivot_aligned = align_htf_to_ltf(prices, df_12h, pivot_val)
            
            if position == 1:
                # Exit long: price returns to monthly pivot OR trend turns down
                if (close[i] <= pivot_aligned[i]) or \
                   (close[i] < ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to monthly pivot OR trend turns up
                if (close[i] >= pivot_aligned[i]) or \
                   (close[i] > ema_50_12h_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals