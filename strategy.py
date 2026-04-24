#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 12h for EMA50 trend direction.
- EMA50 > EMA50_prev indicates bullish trend, EMA50 < EMA50_prev indicates bearish trend.
- Entry: Long when price breaks above Donchian(20) upper AND 12h EMA50 rising (bullish breakout in uptrend).
         Short when price breaks below Donchian(20) lower AND 12h EMA50 falling (bearish breakout in downtrend).
         In ranging (EMA50 flat): Long when price touches Donchian lower AND reverses up (close > low).
                                  Short when price touches Donchian upper AND reverses down (close < high).
- Exit: Opposite Donchian breakout or EMA50 trend reversal.
- Volume confirmation: current volume > 1.3 * 20-period volume MA (to avoid false breakouts).
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
    
    # Get 12h data for EMA50 trend
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 12h
    ema50 = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 12h EMA50 to 4h
    ema50_aligned = align_htf_to_ltf(prices, df_12h, ema50)
    
    # Donchian channels (20-period) on 4h
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: current volume > 1.3 * 20-period volume MA (on 4h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.3 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, lookback, 20)  # Need enough 12h bars for EMA50 and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        # Check EMA50 trend direction (rising/falling/flat)
        if i > start_idx:
            ema50_prev = ema50_aligned[i-1]
            ema50_curr = ema50_aligned[i]
            ema50_rising = ema50_curr > ema50_prev
            ema50_falling = ema50_curr < ema50_prev
            ema50_flat = abs(ema50_curr - ema50_prev) < 1e-10
        else:
            ema50_rising = ema50_falling = ema50_flat = False
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if ema50_rising:  # Bullish trend: breakout strategy
                    # Bullish breakout: price closes above upper Donchian
                    if curr_close > highest_high[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema50_falling:  # Bearish trend: breakout strategy
                    # Bearish breakout: price closes below lower Donchian
                    if curr_close < lowest_low[i]:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging (EMA50 flat): mean reversion at extremes
                    # Long when price touches lower Donchian and shows reversal (close > low)
                    if curr_low <= lowest_low[i] and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches upper Donchian and shows reversal (close < high)
                    elif curr_high >= highest_high[i] and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below Donchian mid OR EMA50 trend turns bearish
            if curr_close < donchian_mid[i] or ema50_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian mid OR EMA50 trend turns bullish
            if curr_close > donchian_mid[i] or ema50_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_12hEMA50Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0