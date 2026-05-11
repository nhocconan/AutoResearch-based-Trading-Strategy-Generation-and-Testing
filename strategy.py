#!/usr/bin/env python3
# 4h_Pivot_Breakout_Supertrend_1d
# Hypothesis: Breakout above daily R1 or below daily S1 with Supertrend (1d) filter on 4h timeframe. Uses daily pivot levels as key support/resistance. Supertrend on daily timeframe filters for trend direction: only take long breaks above R1 when daily trend is up, short breaks below S1 when daily trend is down. Volume spike confirms institutional interest. Designed for 4h with daily pivot structure and daily Supertrend filter to reduce trades and increase win rate. Works in both bull (breakouts in uptrend) and bear (breakdowns in downtrend) by capturing momentum after breaking key daily levels with trend alignment.

name = "4h_Pivot_Breakout_Supertrend_1d"
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
    
    # Daily data for pivot points and Supertrend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 10:
        return np.zeros(n)
    
    # Calculate daily pivot points using previous day's OHLC
    d_high = df_1d['high'].values
    d_low = df_1d['low'].values
    d_close = df_1d['close'].values
    
    # Shift to use previous day's data (avoid look-ahead)
    d_high_prev = np.roll(d_high, 1)
    d_low_prev = np.roll(d_low, 1)
    d_close_prev = np.roll(d_close, 1)
    # First period: use current values to avoid NaN
    d_high_prev[0] = d_high[0]
    d_low_prev[0] = d_low[0]
    d_close_prev[0] = d_close[0]
    
    # Calculate daily pivot point
    d_pivot = (d_high_prev + d_low_prev + d_close_prev) / 3.0
    # Calculate daily R1 and S1 levels (1 * (H-L) from pivot)
    d_r1 = d_pivot + (d_high_prev - d_low_prev)
    d_s1 = d_pivot - (d_high_prev - d_low_prev)
    
    # Align daily R1/S1 to 4h timeframe
    d_r1_aligned = align_htf_to_ltf(prices, df_1d, d_r1)
    d_s1_aligned = align_htf_to_ltf(prices, df_1d, d_s1)
    d_pivot_aligned = align_htf_to_ltf(prices, df_1d, d_pivot)
    
    # Calculate daily Supertrend (ATR=10, multiplier=3.0)
    d_high_close = np.maximum(d_high, np.roll(d_close, 1))
    d_low_close = np.minimum(d_low, np.roll(d_close, 1))
    d_tr = np.maximum(d_high - d_low, np.maximum(d_high_close, d_low_close))
    d_atr = pd.Series(d_tr).ewm(alpha=1/10, adjust=False).mean().values  # Wilder's smoothing
    d_upper = (d_high + d_low) / 2 + 3.0 * d_atr
    d_lower = (d_high + d_low) / 2 - 3.0 * d_atr
    d_upper = np.where(d_close_prev > np.roll(d_upper, 1), d_upper, np.roll(d_upper, 1))
    d_lower = np.where(d_close_prev < np.roll(d_lower, 1), d_lower, np.roll(d_lower, 1))
    d_supertrend = np.where(d_close > d_upper, d_lower,
                           np.where(d_close < d_lower, d_upper, np.roll(d_supertrend, 1) if 'd_supertrend' in locals() else d_upper))
    # Initialize and compute properly
    d_supertrend = np.zeros_like(d_close)
    d_supertrend[0] = d_upper[0]
    for i in range(1, len(d_close)):
        d_upper[i] = (d_high[i] + d_low[i]) / 2 + 3.0 * d_atr[i]
        d_lower[i] = (d_high[i] + d_low[i]) / 2 - 3.0 * d_atr[i]
        if d_close[i-1] > d_supertrend[i-1]:
            d_upper[i] = min(d_upper[i], d_supertrend[i-1])
        else:
            d_lower[i] = max(d_lower[i], d_supertrend[i-1])
        if d_close[i] > d_upper[i]:
            d_supertrend[i] = d_lower[i]
        elif d_close[i] < d_lower[i]:
            d_supertrend[i] = d_upper[i]
        else:
            d_supertrend[i] = d_supertrend[i-1]
    
    # Align daily Supertrend to 4h timeframe
    d_supertrend_aligned = align_htf_to_ltf(prices, df_1d, d_supertrend)
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(d_r1_aligned[i]) or np.isnan(d_s1_aligned[i]) or 
            np.isnan(d_supertrend_aligned[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume threshold
        volume_spike = vol_ratio[i] > 1.5
        
        if position == 0:
            # Long: break above daily R1 + daily Supertrend up (uptrend) + volume spike
            if (close[i] > d_r1_aligned[i] and 
                close[i] > d_supertrend_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below daily S1 + daily Supertrend down (downtrend) + volume spike
            elif (close[i] < d_s1_aligned[i] and 
                  close[i] < d_supertrend_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to daily pivot or Supertrend reversal
            if position == 1:
                # Exit long: price returns to daily pivot OR Supertrend turns down
                if (close[i] <= d_pivot_aligned[i]) or \
                   (close[i] < d_supertrend_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to daily pivot OR Supertrend turns up
                if (close[i] >= d_pivot_aligned[i]) or \
                   (close[i] > d_supertrend_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals