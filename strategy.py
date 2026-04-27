#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-day Williams %R with 1-week EMA trend filter and volume confirmation.
# Williams %R identifies overbought/oversold conditions: > -20 = overbought, < -80 = oversold.
# Strategy: In ranging markets, buy oversold (< -80) and sell overbought (> -20).
# In trending markets, only take trades in direction of weekly EMA trend.
# Volume spike confirms institutional participation.
# Designed for ~15-25 trades/year per symbol to avoid fee drag.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams %R calculation (14-period)
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    
    williams_r = -100 * (highest_high - close) / hl_range
    
    # Get 1-week data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    # 34-period EMA on 1w close for trend filter
    ema34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(williams_r[i]) or np.isnan(ema34_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Williams %R signals
        if williams_r[i] < -80:  # Oversold - potential long
            if close[i] > ema34_1w_aligned[i] and volume_filter[i]:  # Only long in uptrend
                signals[i] = 0.25
                position = 1
            elif close[i] <= ema34_1w_aligned[i]:  # In downtrend, wait for pullback
                if position == 1:
                    signals[i] = 0.25  # Hold long
                else:
                    signals[i] = 0.0
        elif williams_r[i] > -20:  # Overbought - potential short
            if close[i] < ema34_1w_aligned[i] and volume_filter[i]:  # Only short in downtrend
                signals[i] = -0.25
                position = -1
            elif close[i] >= ema34_1w_aligned[i]:  # In uptrend, wait for rally
                if position == -1:
                    signals[i] = -0.25  # Hold short
                else:
                    signals[i] = 0.0
        else:  # Neutral zone (-80 to -20)
            # Hold existing position if any
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "1d_WilliamsR_1wEMA34_VolumeFilter"
timeframe = "1d"
leverage = 1.0