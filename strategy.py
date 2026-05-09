#!/usr/bin/env python3
# 4H_1D_Poisson_Rainbow_Volume
# Hypothesis: Combines Poisson-based trend strength (count of closes above/below EMA) with
# rainbow EMA convergence/divergence and volume spikes to capture sustained moves.
# Works in bull/bear by requiring both trend alignment and volatility expansion.
# Target: 20-50 trades/year per symbol (80-200 total over 4 years).

name = "4H_1D_Poisson_Rainbow_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Poisson trend strength: count of closes above/below EMA(21) in last 10 days
    ema_21_1d = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    above_ema = (close_1d > ema_21_1d).astype(int)
    below_ema = (close_1d < ema_21_1d).astype(int)
    # Sum over last 10 days: strong trend if >=7 days in same direction
    poisson_long = pd.Series(above_ema).rolling(window=10, min_periods=10).sum().values >= 7
    poisson_short = pd.Series(below_ema).rolling(window=10, min_periods=10).sum().values >= 7
    
    # Rainbow EMA convergence: EMA(8,13,21) - look for compression/expansion
    ema_8 = pd.Series(close_1d).ewm(span=8, adjust=False, min_periods=8).mean().values
    ema_13 = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    ema_21 = pd.Series(close_1d).ewm(span=21, adjust=False, min_periods=21).mean().values
    # Rainbow spread: (EMA8 - EMA21) / EMA21 - measures convergence
    rainbow_spread = (ema_8 - ema_21) / ema_21
    # Converging when spread decreasing, expanding when increasing
    rainbow_converging = np.diff(rainbow_spread, prepend=0) < 0
    rainbow_expanding = np.diff(rainbow_spread, prepend=0) > 0
    
    # Volume spike: current > 2.0x 24-period average (1 day of 4h bars)
    volume_avg = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_spike = volume > (volume_avg * 2.0)
    
    # Align 1d indicators to 4h
    poisson_long_aligned = align_htf_to_ltf(prices, df_1d, poisson_long)
    poisson_short_aligned = align_htf_to_ltf(prices, df_1d, poisson_short)
    rainbow_converging_aligned = align_htf_to_ltf(prices, df_1d, rainbow_converging)
    rainbow_expanding_aligned = align_htf_to_ltf(prices, df_1d, rainbow_expanding)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)  # though volume is LTF, align for consistency
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after we have enough data
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(poisson_long_aligned[i]) or np.isnan(poisson_short_aligned[i]) or
            np.isnan(rainbow_converging_aligned[i]) or np.isnan(rainbow_expanding_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: strong uptrend (Poisson) + expanding rainbow + volume spike
            if poisson_long_aligned[i] and rainbow_expanding_aligned[i] and volume_spike_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: strong downtrend (Poisson) + expanding rainbow + volume spike
            elif poisson_short_aligned[i] and rainbow_expanding_aligned[i] and volume_spike_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: trend weakening (Poisson fails) OR rainbow converging (loss of momentum)
            if not poisson_long_aligned[i] or rainbow_converging_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: trend weakening (Poisson fails) OR rainbow converging
            if not poisson_short_aligned[i] or rainbow_converging_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals