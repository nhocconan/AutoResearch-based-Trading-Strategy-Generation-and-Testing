#!/usr/bin/env python3
"""
1d_Adaptive_Kelly_Donchian_20_1wTrend_Filter
Hypothesis: Daily Donchian(20) breakouts in direction of weekly trend with adaptive Kelly position sizing.
Uses weekly EMA50 as trend filter to avoid counter-trend trades. Adaptive Kelly scales position by volatility
(inverse ATR) to maintain consistent risk. Designed for low trade frequency (~10-20/year) to work in both
bull and bear markets via trend alignment and volatility-adjusted sizing. Donchian breakouts capture strong
momentum shifts with clear entry/exit rules.
"""

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
    
    # Get 1w data for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA50 on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w, additional_delay_bars=1)
    
    # Calculate ATR(14) for volatility normalization
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr2])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Donchian channels (20-period)
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for ATR(14) and Donchian(20)
    start_idx = max(20, 14)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_1w_aligned[i]) or 
            np.isnan(atr[i]) or atr[i] == 0 or
            np.isnan(highest_high[i]) or
            np.isnan(lowest_low[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Base Kelly fraction (conservative)
        base_kelly = 0.15
        
        # Volatility scaling (inverse ATR normalized to median)
        if i >= 50:
            vol_scaling = np.median(atr[i-50:i]) / atr[i]
            vol_scaling = np.clip(vol_scaling, 0.5, 2.0)  # Limit scaling
        else:
            vol_scaling = 1.0
        
        # Adaptive position size
        position_size = base_kelly * vol_scaling
        position_size = min(position_size, 0.35)  # Cap at 35%
        
        if position == 0:
            # Look for Donchian breakout signals with weekly trend filter
            # Long: price breaks above 20-period high in uptrend (close > weekly EMA50)
            # Short: price breaks below 20-period low in downtrend (close < weekly EMA50)
            long_signal = (close[i] > highest_high[i]) and (close[i] > ema50_1w_aligned[i])
            short_signal = (close[i] < lowest_low[i]) and (close[i] < ema50_1w_aligned[i])
            
            if long_signal:
                signals[i] = position_size
                position = 1
            elif short_signal:
                signals[i] = -position_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = position_size
            # Exit when price moves back below 20-period low (mean reversion)
            exit_signal = close[i] < lowest_low[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -position_size
            # Exit when price moves back above 20-period high (mean reversion)
            exit_signal = close[i] > highest_high[i]
            if exit_signal:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_Adaptive_Kelly_Donchian_20_1wTrend_Filter"
timeframe = "1d"
leverage = 1.0