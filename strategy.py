#!/usr/bin/env python3
# 6h_1w1d_RVOL_Momentum_Breakout
# Hypothesis: 6h breakouts with volume surge (1.5x 24-period avg) and 1w/1d trend alignment.
# Uses 1w EMA10 for long-term bias and 1d EMA20 for medium-term trend.
# Volume surge confirms institutional participation, reducing false breakouts.
# Designed for 6h timeframe to target 12-37 trades/year per symbol.
# Works in bull/bear by requiring trend alignment, avoiding chop whipsaws.

name = "6h_1w1d_RVOL_Momentum_Breakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d and 1w data for trend and momentum
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1d) < 20 or len(df_1w) < 10:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1w EMA10 for long-term trend
    ema_10_1w = pd.Series(df_1w['close'].values).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate 1d EMA20 for medium-term trend
    ema_20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 6-period volume average (6 periods * 6h = 36h ~ 1.5 days)
    vol_ma = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    # Align HTF indicators to 6h timeframe
    ema_10_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_10_1w)
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    close_1d_aligned = align_htf_to_ltf(prices, df_1d, df_1d['close'].values)
    close_1w_aligned = align_htf_to_ltf(prices, df_1w, df_1w['close'].values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for EMA20 and volume MA
    start_idx = 20
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_10_1w_aligned[i]) or
            np.isnan(ema_20_1d_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend alignment: 1w up AND 1d up for long, 1w down AND 1d down for short
        uptrend_aligned = close_1w_aligned[i] > ema_10_1w_aligned[i] and close_1d_aligned[i] > ema_20_1d_aligned[i]
        downtrend_aligned = close_1w_aligned[i] < ema_10_1w_aligned[i] and close_1d_aligned[i] < ema_20_1d_aligned[i]
        
        # Volume confirmation (1.5x average for significance)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Breakout with volume surge in aligned uptrend
            if close[i] > high[i-1] and uptrend_aligned and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown with volume surge in aligned downtrend
            elif close[i] < low[i-1] and downtrend_aligned and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: close below previous low or trend fails
                if close[i] < low[i-1] or not uptrend_aligned:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: close above previous high or trend fails
                if close[i] > high[i-1] or not downtrend_aligned:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals