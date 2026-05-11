#!/usr/bin/env python3
"""
1d_WeeklyPivot_R3S3_Breakout_TrendVolume_v2
Hypothesis: Price breaking above weekly R3 or below weekly S3 with 1d trend confirmation (EMA34) and volume spike. Uses weekly pivot levels as strong support/resistance. In uptrend (price > EMA34), buy breakouts above R3; in downtrend (price < EMA34), sell breakdowns below S3. Volume confirms institutional interest. Designed for 1d timeframe with 1d trend filter and weekly pivots to reduce trades and increase win rate. Works in both bull (breakouts) and bear (breakdowns) markets.
"""

name = "1d_WeeklyPivot_R3S3_Breakout_TrendVolume_v2"
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
    # Using previous period's OHLC
    w_high = df_1w['high'].values
    w_low = df_1w['low'].values
    w_close = df_1w['close'].values
    
    # Shift to use previous period's data (avoid look-ahead)
    w_high_prev = np.roll(w_high, 1)
    w_low_prev = np.roll(w_low, 1)
    w_close_prev = np.roll(w_close, 1)
    # First period: use current values to avoid NaN
    w_high_prev[0] = w_high[0]
    w_low_prev[0] = w_low[0]
    w_close_prev[0] = w_close[0]
    
    # Pivot point calculation
    pivot = (w_high_prev + w_low_prev + w_close_prev) / 3.0
    # R3 and S3 levels
    r3 = pivot + 3 * (w_high_prev - w_low_prev)
    s3 = pivot - 3 * (w_high_prev - w_low_prev)
    
    # Align weekly R3/S3 to 1d timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1w, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1w, s3)
    
    # 1d trend filter (EMA 34)
    ema_34_1d = pd.Series(close).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    
    # Volume confirmation (20-period average on 1d)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_1d[i]) or np.isnan(vol_ratio[i])):
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
            # Long: break above weekly R3 + above EMA34 + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_34_1d[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below weekly S3 + below EMA34 + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_34_1d[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to pivot or trend reversal
            # Calculate pivot for exit (use previous period's data)
            pivot_val = (w_high_prev + w_low_prev + w_close_prev) / 3.0
            pivot_aligned = align_htf_to_ltf(prices, df_1w, pivot_val)
            
            if position == 1:
                # Exit long: price returns to weekly pivot OR trend turns down
                if (close[i] <= pivot_aligned[i]) or \
                   (close[i] < ema_34_1d[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to weekly pivot OR trend turns up
                if (close[i] >= pivot_aligned[i]) or \
                   (close[i] > ema_34_1d[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals