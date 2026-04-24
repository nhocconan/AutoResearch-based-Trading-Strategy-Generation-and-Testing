#!/usr/bin/env python3
"""
Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 1d for execution, HTF: 1w for EMA50 trend direction.
- EMA50 > rising indicates bullish trend; EMA50 < falling indicates bearish trend.
- Entry: Long when price breaks above Donchian(20) upper AND 1w EMA50 is rising.
         Short when price breaks below Donchian(20) lower AND 1w EMA50 is falling.
         In ranging (1w EMA50 flat): mean reversion at Donchian extremes.
- Exit: Opposite Donchian breakout or 1w EMA50 trend reversal.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 30-100 total trades over 4 years (7-25/year) for 1d timeframe.
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
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w
    ema50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_prev = pd.Series(ema50).shift(1).values  # Previous bar EMA50 for slope
    ema50_rising = ema50 > ema50_prev  # Bullish trend
    ema50_falling = ema50 < ema50_prev  # Bearish trend
    
    # Align 1w EMA50 and trend to 1d
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    ema50_rising_aligned = align_htf_to_ltf(prices, df_1w, ema50_rising.astype(float))
    ema50_falling_aligned = align_htf_to_ltf(prices, df_1w, ema50_falling.astype(float))
    
    # Donchian channels (20-period) on 1d
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    donchian_mid = (highest_high + lowest_low) / 2.0
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 1d)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, lookback, 20)  # Need enough 1w bars for EMA50 and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(ema50_rising_aligned[i]) or np.isnan(ema50_falling_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        ema50_val = ema50_aligned[i]
        ema50_rising_val = bool(ema50_rising_aligned[i])
        ema50_falling_val = bool(ema50_falling_aligned[i])
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if ema50_rising_val:  # Bullish trend: breakout strategy
                    # Bullish breakout: price closes above upper Donchian
                    if curr_close > highest_high[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema50_falling_val:  # Bearish trend: breakout strategy
                    # Bearish breakout: price closes below lower Donchian
                    if curr_close < lowest_low[i]:
                        signals[i] = -0.25
                        position = -1
                else:  # Ranging trend (EMA50 flat): mean reversion at extremes
                    # Long when price touches lower Donchian and shows reversal (close > low)
                    if curr_low <= lowest_low[i] and curr_close > curr_low:
                        signals[i] = 0.25
                        position = 1
                    # Short when price touches upper Donchian and shows reversal (close < high)
                    elif curr_high >= highest_high[i] and curr_close < curr_high:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below Donchian mid OR EMA50 turns bearish
            if curr_close < donchian_mid[i] or ema50_falling_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian mid OR EMA50 turns bullish
            if curr_close > donchian_mid[i] or ema50_rising_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_Donchian20_1wEMA50Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0