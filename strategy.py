#!/usr/bin/env python3
"""
6h_Williams_VIX_Fix_Confluence_v1
Hypothesis: Combines Williams VIX Fix (volatility spike detector) with 12h EMA trend filter and 6h Donchian breakout for entries.
Long when: VIX Fix > 0.8 (high fear), price > 12h EMA50, and break above 6h Donchian(20) high.
Short when: VIX Fix > 0.8 (high fear), price < 12h EMA50, and break below 6h Donchian(20) low.
Exit when VIX Fix < 0.3 (low fear) or opposite Donchian breakout.
Designed to catch panic-driven reversals in both bull and bear markets with tight entries.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn. Targets 80-120 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Williams VIX Fix: measures fear/greed based on price range relative to recent high
    # VIX Fix = (Highest Close in lookback - Low) / (Highest Close in lookback - Lowest Close in lookback) * 100
    lookback = 22
    highest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).max().values
    lowest_close = pd.Series(close).rolling(window=lookback, min_periods=lookback).min().values
    vix_fix = (highest_close - low) / (highest_close - lowest_close + 1e-10)
    
    # Load 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # 6h Donchian channels for breakout signals
    donchian_lookback = 20
    donchian_high = pd.Series(high).rolling(window=donchian_lookback, min_periods=donchian_lookback).max().values
    donchian_low = pd.Series(low).rolling(window=donchian_lookback, min_periods=donchian_lookback).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = max(lookback, donchian_lookback, 50) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(vix_fix[i]) or np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
            continue
        
        # Long logic: high fear (VIX Fix > 0.8), above 12h EMA50, and Donchian breakout
        if (vix_fix[i] > 0.8 and 
            close[i] > ema_50_12h_aligned[i] and 
            high[i] > donchian_high[i]):
            if position != 1:
                signals[i] = 0.25
                position = 1
            else:
                signals[i] = 0.25
        # Short logic: high fear (VIX Fix > 0.8), below 12h EMA50, and Donchian breakout down
        elif (vix_fix[i] > 0.8 and 
              close[i] < ema_50_12h_aligned[i] and 
              low[i] < donchian_low[i]):
            if position != -1:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = -0.25
        # Exit conditions: low fear (VIX Fix < 0.3) or opposite Donchian breakout
        elif position == 1 and (vix_fix[i] < 0.3 or low[i] < donchian_low[i]):
            signals[i] = 0.0
            position = 0
        elif position == -1 and (vix_fix[i] < 0.3 or high[i] > donchian_high[i]):
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.25
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Williams_VIX_Fix_Confluence_v1"
timeframe = "6h"
leverage = 1.0