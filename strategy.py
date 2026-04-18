#!/usr/bin/env python3
"""
12h_Weekly_Momentum_Breakout
Hypothesis: Uses weekly momentum (price above weekly EMA20) to establish trend direction, then enters on 12h price breaking above/below the 20-period Donchian channel with volume confirmation. Designed to capture strong trending moves while minimizing false signals in ranging markets. Weekly trend filter reduces whipsaws in bear markets, volume confirmation ensures institutional participation, and Donchian breakout captures momentum bursts. Targets 15-30 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly EMA20 for trend filter
    close_1w = df_1w['close'].values
    ema_20_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 20:
        k = 2 / (20 + 1)
        ema_20_1w[19] = np.mean(close_1w[:20])
        for i in range(20, len(close_1w)):
            ema_20_1w[i] = close_1w[i] * k + ema_20_1w[i-1] * (1 - k)
    
    # Align weekly EMA20 to 12h timeframe
    ema_20_aligned = align_htf_to_ltf(prices, df_1w, ema_20_1w)
    
    # Calculate 12h Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    start_idx = 40  # Warmup for Donchian and weekly EMA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_20_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        bars_since_entry += 1
        
        if position == 0:
            # Long: price above weekly EMA20 (uptrend) + breaks above Donchian high + volume spike
            if close[i] > ema_20_aligned[i] and close[i] > donchian_high[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            # Short: price below weekly EMA20 (downtrend) + breaks below Donchian low + volume spike
            elif close[i] < ema_20_aligned[i] and close[i] < donchian_low[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
        
        elif position == 1:
            # Exit: minimum 4 bars hold, then exit on trend reversal or volatility drop
            if bars_since_entry >= 4:
                if close[i] < ema_20_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:
                signals[i] = 0.25  # Hold during minimum period
        
        elif position == -1:
            # Exit: minimum 4 bars hold, then exit on trend reversal or volatility drop
            if bars_since_entry >= 4:
                if close[i] > ema_20_aligned[i] or not vol_spike[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
            else:
                signals[i] = -0.25  # Hold during minimum period
    
    return signals

name = "12h_Weekly_Momentum_Breakout"
timeframe = "12h"
leverage = 1.0