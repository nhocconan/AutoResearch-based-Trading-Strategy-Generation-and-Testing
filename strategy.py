#!/usr/bin/env python3
# 6h_1w_donchian_pivot_v1
# Strategy: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Weekly pivots define institutional support/resistance. Donchian breakouts in the
# direction of weekly pivot bias capture trend continuation with institutional backing.
# Volume filter ensures breakout authenticity. Designed for low frequency (15-25 trades/year)
# to minimize fee drag in all market regimes.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_donchian_pivot_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*P - L, S1 = 2*P - H
    # R2 = P + (H - L), S2 = P - (H - L)
    # R3 = H + 2*(P - L), S3 = L - 2*(H - P)
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    weekly_close = df_1w['close'].values
    
    pivot = (weekly_high + weekly_low + weekly_close) / 3.0
    r1 = 2 * pivot - weekly_low
    s1 = 2 * pivot - weekly_high
    r2 = pivot + (weekly_high - weekly_low)
    s2 = pivot - (weekly_high - weekly_low)
    r3 = weekly_high + 2 * (pivot - weekly_low)
    s3 = weekly_low - 2 * (weekly_high - pivot)
    
    # Weekly bias: price above/below pivot
    weekly_bias_up = weekly_close > pivot
    weekly_bias_down = weekly_close < pivot
    
    # Align weekly bias to 6h timeframe
    weekly_bias_up_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_up)
    weekly_bias_down_aligned = align_htf_to_ltf(prices, df_1w, weekly_bias_down)
    
    # Donchian channels (20-period)
    lookback = 20
    donchian_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    donchian_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    volume_spike = volume > (vol_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup period
        # Skip if any required data is invalid
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(weekly_bias_up_aligned[i]) or np.isnan(weekly_bias_down_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Donchian breakout conditions
        breakout_up = close[i] > donchian_high[i-1]  # Break above previous high
        breakout_down = close[i] < donchian_low[i-1]  # Break below previous low
        
        # Weekly bias
        bias_up = weekly_bias_up_aligned[i]
        bias_down = weekly_bias_down_aligned[i]
        
        # Entry logic: Donchian breakout in direction of weekly bias + volume spike
        if (breakout_up and bias_up and volume_spike[i] and position != 1):
            position = 1
            signals[i] = 0.25
        elif (breakout_down and bias_down and volume_spike[i] and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: opposite breakout or loss of weekly bias
        elif position == 1 and (breakout_down or not bias_up):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (breakout_up or not bias_down):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals