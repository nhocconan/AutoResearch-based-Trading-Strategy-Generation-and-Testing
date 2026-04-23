#!/usr/bin/env python3
"""
Hypothesis: 6h Williams %R + 12h Supertrend combination for mean reversion in ranging markets with trend filter.
Long when Williams %R(14) crosses above -80 (oversold) AND 12h Supertrend is bullish.
Short when Williams %R(14) crosses below -20 (overbought) AND 12h Supertrend is bearish.
Exit when Williams %R returns to -50 (mean reversion target) or Supertrend flips.
Designed for ~15-25 trades/year with mean reversion edge in choppy markets + trend filter to avoid whipsaws.
Williams %R identifies exhaustion points; 12h Supertrend ensures higher timeframe alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Calculate 12h Supertrend for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Supertrend parameters
    atr_period = 10
    multiplier = 3.0
    
    # Calculate ATR
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.max([high_12h[0] - low_12h[0], np.abs(high_12h[0] - close_12h[0]), np.abs(low_12h[0] - close_12h[0])])], 
                         np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Calculate basic upper and lower bands
    hl2 = (high_12h + low_12h) / 2
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_12h, np.nan, dtype=float)
    direction = np.full_like(close_12h, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = upper_band[0] if not np.isnan(upper_band[0]) else close_12h[0]
    direction[0] = 1
    
    for i in range(1, len(close_12h)):
        if np.isnan(atr[i]) or np.isnan(upper_band[i]) or np.isnan(lower_band[i]):
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            continue
            
        if close_12h[i] > supertrend[i-1]:
            direction[i] = 1
        elif close_12h[i] < supertrend[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        
        if direction[i] == 1:
            supertrend[i] = max(lower_band[i], supertrend[i-1])
        else:
            supertrend[i] = min(upper_band[i], supertrend[i-1])
    
    # Align Supertrend to 6h timeframe
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, supertrend)
    direction_aligned = align_htf_to_ltf(prices, df_12h, direction.astype(float))
    
    # Calculate Williams %R on 6h data
    lookback = 14
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Avoid division by zero
    hl_range = highest_high - lowest_low
    hl_range = np.where(hl_range == 0, 1e-10, hl_range)
    williams_r = -100 * ((highest_high - close) / hl_range)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(lookback, atr_period)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(supertrend_aligned[i]) or np.isnan(direction_aligned[i]) or 
            np.isnan(williams_r[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Williams %R conditions
        wr = williams_r[i]
        wr_prev = williams_r[i-1]
        
        # Cross above -80 (oversold)
        cross_above_80 = (wr_prev <= -80) and (wr > -80)
        # Cross below -20 (overbought)
        cross_below_20 = (wr_prev >= -20) and (wr < -20)
        # Return to -50 (mean reversion target)
        return_to_50 = (position == 1 and wr >= -50) or (position == -1 and wr <= -50)
        
        # Trend filter: Supertrend direction
        trend_up = direction_aligned[i] == 1
        trend_down = direction_aligned[i] == -1
        
        if position == 0:
            # Long: Williams %R crosses above -80 AND 12h Supertrend bullish
            if cross_above_80 and trend_up:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 AND 12h Supertrend bearish
            elif cross_below_20 and trend_down:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: return to -50 or Supertrend flip
            exit_signal = False
            if position == 1:
                exit_signal = return_to_50 or (direction_aligned[i] == -1)
            elif position == -1:
                exit_signal = return_to_50 or (direction_aligned[i] == 1)
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6H_WilliamsR_Supertrend_Combo_12hTF"
timeframe = "6h"
leverage = 1.0