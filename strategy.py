#!/usr/bin/env python3
"""
4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn
Hypothesis: Price breaking above daily R1 or below daily S1 with 1d EMA34 trend confirmation and volume spike.
Uses daily pivot levels as strong support/resistance. In uptrend (price > EMA34 on 1d), buy breakouts above R1;
in downtrend (price < EMA34 on 1d), sell breakdowns below S1. Volume confirms institutional interest.
Designed for 4h timeframe with daily pivot structure and 1d trend filter to reduce trades and increase win rate.
Works in both bull (breakouts) and bear (breakdowns) markets by capturing strong momentum moves after breaking key daily levels.
"""

name = "4h_Camarilla_R1_S1_Breakout_1dEMA34_VolumeSpike_Dyn"
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
    
    # Daily data for pivot points and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
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
    # Calculate daily R1 and S1 levels
    d_r1 = d_pivot + (1.1/2) * (d_high_prev - d_low_prev)
    d_s1 = d_pivot - (1.1/2) * (d_high_prev - d_low_prev)
    
    # Align daily R1/S1 to 4h timeframe
    d_r1_aligned = align_htf_to_ltf(prices, df_1d, d_r1)
    d_s1_aligned = align_htf_to_ltf(prices, df_1d, d_s1)
    d_pivot_aligned = align_htf_to_ltf(prices, df_1d, d_pivot)
    
    # 1d trend filter (EMA 34)
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(
        span=34, adjust=False, min_periods=34
    ).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume confirmation (20-period average on 4h)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 40
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(d_r1_aligned[i]) or np.isnan(d_s1_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ratio[i])):
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
            # Long: break above daily R1 + above 1d EMA34 + volume spike
            if (close[i] > d_r1_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike):
                signals[i] = 0.25
                position = 1
            # Short: break below daily S1 + below 1d EMA34 + volume spike
            elif (close[i] < d_s1_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to daily pivot or trend reversal
            if position == 1:
                # Exit long: price returns to daily pivot OR trend turns down
                if (close[i] <= d_pivot_aligned[i]) or \
                   (close[i] < ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price returns to daily pivot OR trend turns up
                if (close[i] >= d_pivot_aligned[i]) or \
                   (close[i] > ema_34_1d_aligned[i]):
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals