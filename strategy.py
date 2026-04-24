#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for EMA50 trend direction.
- EMA50 > rising: bullish trend, only allow long breakouts.
- EMA50 < falling: bearish trend, only allow short breakouts.
- EMA50 flat: ranging, no new entries (reduce false breakouts).
- Entry: Long when price closes above Donchian(20) upper AND 12h EMA50 trending up.
         Short when price closes below Donchian(20) lower AND 12h EMA50 trending down.
- Exit: Opposite Donchian breakout or EMA50 flattening.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 75-200 total trades over 4 years (19-50/year) for 4h timeframe.
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
    ema_50 = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    # Calculate EMA50 slope for trend detection (using 3-bar change)
    ema_slope = np.zeros_like(ema_50_aligned)
    ema_slope[3:] = (ema_50_aligned[3:] - ema_50_aligned[:-3]) / 3
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50 + 3, lookback, 20)  # Need enough 12h bars for EMA50 and slope + lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_slope[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Bullish breakout: price closes above upper Donchian AND EMA50 trending up
                if curr_close > highest_high[i] and ema_slope[i] > 0:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakout: price closes below lower Donchian AND EMA50 trending down
                elif curr_close < lowest_low[i] and ema_slope[i] < 0:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below Donchian mid OR EMA50 flattens/down
            if curr_close < donchian_mid[i] or ema_slope[i] <= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian mid OR EMA50 flattens/up
            if curr_close > donchian_mid[i] or ema_slope[i] >= 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0