#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian(20) breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 4h for execution, HTF: 1w for EMA50 trend strength.
- EMA50 > rising indicates bullish trend (breakout strategy), EMA50 < falling indicates bearish trend.
- Entry: Long when price breaks above Donchian(20) upper AND 1w EMA50 rising (bullish breakout in uptrend).
         Short when price breaks below Donchian(20) lower AND 1w EMA50 falling (bearish breakout in downtrend).
- Exit: Opposite Donchian breakout or 1w EMA50 flattening (slope near zero).
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
    
    # Get 1w data for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w
    ema50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 4h
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    
    # Calculate EMA50 slope (rising/falling) on 1w
    ema50_slope = np.zeros_like(ema50)
    ema50_slope[1:] = (ema50[1:] - ema50[:-1]) / ema50[:-1]  # percentage change
    ema50_slope_aligned = align_htf_to_ltf(prices, df_1w, ema50_slope)
    
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
    start_idx = max(50, lookback, 20)  # Need enough 1w bars for EMA50 and lookback for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(ema50_slope_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_aligned[i]
        ema50_slope_val = ema50_slope_aligned[i]
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                if ema50_slope_val > 0.001:  # Rising EMA50: bullish trend
                    # Bullish breakout: price closes above upper Donchian
                    if curr_close > highest_high[i]:
                        signals[i] = 0.25
                        position = 1
                elif ema50_slope_val < -0.001:  # Falling EMA50: bearish trend
                    # Bearish breakout: price closes below lower Donchian
                    if curr_close < lowest_low[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below Donchian mid OR EMA50 flattens
            if curr_close < donchian_mid[i] or abs(ema50_slope_val) <= 0.001:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above Donchian mid OR EMA50 flattens
            if curr_close > donchian_mid[i] or abs(ema50_slope_val) <= 0.001:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1wEMA50Trend_VolumeConfirm_v1"
timeframe = "4h"
leverage = 1.0