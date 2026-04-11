#!/usr/bin/env python3
# 6h_1w_1d_ema_pullback_v1
# Strategy: 60-day EMA pullback on 6h timeframe with 1-week EMA trend filter and volume confirmation
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: In trending markets (as defined by weekly EMA), price pulls back to the 60-period EMA on 6h chart
# provide high-probability entries. Volume confirmation ensures institutional participation. This strategy
# works in both bull and bear markets by trading with the higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_ema_pullback_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # 60-period EMA on 6h
    ema_60 = pd.Series(close).ewm(span=60, adjust=False, min_periods=60).mean().values
    
    # 200-period EMA on 1d for trend filter
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 50-period EMA on 1w for super trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # 20-period volume average on 1d for confirmation
    volume_1d = df_1d['volume'].values
    vol_avg_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_20_1d)
    
    # Align raw 1d volume for confirmation
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(60, n):
        # Skip if any required data is invalid
        if np.isnan(ema_60[i]) or np.isnan(ema_200_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or \
           np.isnan(vol_avg_20_1d_aligned[i]) or np.isnan(vol_1d_aligned[i]):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Volume confirmation: current 1d volume > 1.3x 20-period average
        vol_confirm = vol_1d_aligned[i] > 1.3 * vol_avg_20_1d_aligned[i]
        
        # Trend filters
        uptrend_1d = close[i] > ema_200_1d_aligned[i]
        uptrend_1w = ema_200_1d_aligned[i] > ema_50_1w_aligned[i]
        downtrend_1d = close[i] < ema_200_1d_aligned[i]
        downtrend_1w = ema_200_1d_aligned[i] < ema_50_1w_aligned[i]
        
        # Price relative to 60 EMA
        near_ema = np.abs(close[i] - ema_60[i]) / ema_60[i] < 0.015  # Within 1.5% of EMA60
        
        # Entry conditions
        # Long: Price near 60 EMA AND uptrend on 1d AND uptrend on 1w AND volume confirmation
        if near_ema and uptrend_1d and uptrend_1w and vol_confirm and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: Price near 60 EMA AND downtrend on 1d AND downtrend on 1w AND volume confirmation
        elif near_ema and downtrend_1d and downtrend_1w and vol_confirm and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: Price moves 2% away from EMA60 (take profit) or trend breaks
        elif position == 1 and (np.abs(close[i] - ema_60[i]) / ema_60[i] > 0.02 or not uptrend_1d):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (np.abs(close[i] - ema_60[i]) / ema_60[i] > 0.02 or not downtrend_1d):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals