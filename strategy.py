#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 12h for EMA50 trend direction.
- EMA50 > rising indicates bullish trend (favor longs), EMA50 < falling indicates bearish trend (favor shorts).
- Entry: Long when price breaks above Donchian(20) upper AND EMA50 trending up AND volume spike.
         Short when price breaks below Donchian(20) lower AND EMA50 trending down AND volume spike.
- Exit: Opposite Donchian breakout or EMA50 trend reversal.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
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
    
    # Get 12h data for EMA50
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA(50) on 12h
    ema_50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 6h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Donchian channels (20-period) on 6h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, lookback, 20)  # Need enough 12h bars for EMA50 and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema_val = ema_50_aligned[i]
        ema_prev = ema_50_aligned[i-1] if i > 0 else ema_val
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Determine EMA trend: up if current > previous, down if current < previous
                ema_trending_up = ema_val > ema_prev
                ema_trending_down = ema_val < ema_prev
                
                # Bullish breakout: price closes above upper Donchian AND EMA trending up
                if curr_close > highest_high[i] and ema_trending_up:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price closes below lower Donchian AND EMA trending down
                elif curr_close < lowest_low[i] and ema_trending_down:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below Donchian mid OR EMA trend turns down
            if curr_close < donchian_mid[i] or ema_val < ema_prev:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian mid OR EMA trend turns up
            if curr_close > donchian_mid[i] or ema_val > ema_prev:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_12hEMA50Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0