#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1w for EMA50 trend direction.
- EMA50 > rising: bullish bias, only take long breakouts above Donchian upper.
- EMA50 < falling: bearish bias, only take short breakouts below Donchian lower.
- EMA50 flat: no trades (avoid chop/whipsaw).
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Works in bull/bear by only trading with the weekly trend, avoiding counter-trend breakouts.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Extract price and volume data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w
    ema_50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    # Donchian channels (20-period) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, lookback, 20)  # Need enough 1w bars for EMA50 and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_50_val = ema_50_aligned[i]
        ema_50_prev = ema_50_aligned[i-1] if i > 0 else ema_50_val
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Determine weekly trend: rising, falling, or flat
        ema_slope = ema_50_val - ema_50_prev
        trend_rising = ema_slope > 0.001 * ema_50_val  # 0.1% threshold
        trend_falling = ema_slope < -0.001 * ema_50_val
        
        if position == 0:
            # Check for entry signals only in trending conditions
            if volume_spike[i]:
                if trend_rising:
                    # Bullish breakout: price closes above upper Donchian
                    if curr_close > highest_high[i]:
                        signals[i] = 0.25
                        position = 1
                elif trend_falling:
                    # Bearish breakout: price closes below lower Donchian
                    if curr_close < lowest_low[i]:
                        signals[i] = -0.25
                        position = -1
                # No trades in flat/ranging market (trend too weak)
        elif position == 1:
            # Long exit: price closes below Donchian mid OR weekly trend turns bearish
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2.0
            if curr_close < donchian_mid or trend_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian mid OR weekly trend turns bullish
            donchian_mid = (highest_high[i] + lowest_low[i]) / 2.0
            if curr_close > donchian_mid or trend_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0