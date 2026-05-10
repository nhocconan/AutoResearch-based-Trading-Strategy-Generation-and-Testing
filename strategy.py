#!/usr/bin/env python3
# 4h_Donchian_Breakout_12hTrend_Volume
# Hypothesis: Breakout above 4h Donchian(20) high with volume surge and 12h EMA50 trend confirmation.
# Works in bull/bear by requiring trend alignment, reducing false breakouts. Targets 20-40 trades/year.

name = "4h_Donchian_Breakout_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    
    # 4h Donchian channel (20-period)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Donchian high/low
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume average (20-period = ~10 days of 4h bars)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0
    
    # Warmup: need Donchian (20) + EMA50 (50) + volume MA (20)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        # Determine trend from 12h EMA50
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, df_12h['close'].values)
        uptrend = close_12h_aligned[i] > ema_50_12h_aligned[i]
        downtrend = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation (1.7x average)
        volume_surge = volume[i] > 1.7 * vol_ma[i]
        
        # Breakout above Donchian high or breakdown below low
        breakout_high = close[i] > donchian_high[i-1]
        breakdown_low = close[i] < donchian_low[i-1]
        
        if position == 0:
            bars_since_entry = 0
            # Long: Breakout above Donchian high with volume surge and 12h uptrend
            if breakout_high and volume_surge and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Donchian low with volume surge and 12h downtrend
            elif breakdown_low and volume_surge and downtrend:
                signals[i] = -0.25
                position = -1
        else:
            bars_since_entry += 1
            # Enforce minimum holding period of 3 bars (12 hours)
            if bars_since_entry < 3:
                signals[i] = signals[i-1]  # maintain position
                continue
            
            if position == 1:
                # Long exit: price breaks below Donchian low or trend changes
                if close[i] < donchian_low[i-1] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price breaks above Donchian high or trend changes
                if close[i] > donchian_high[i-1] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals