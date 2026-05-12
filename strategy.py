#!/usr/bin/env python3
# 6h_Adaptive_Trend_Follow_12hTrend_Volume
# Hypothesis: Use adaptive trend following on 6h timeframe with 12h trend filter and volume confirmation.
# The strategy adapts to volatility by using ATR-based position sizing and trend strength filtering.
# In bull markets: captures trend continuation. In bear markets: avoids false signals via trend filter.
# Volume confirmation ensures moves have participation. Target: 15-35 trades/year to minimize fee drag.

name = "6h_Adaptive_Trend_Follow_12hTrend_Volume"
timeframe = "6h"
leverage = 1.0

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
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 6h indicators
    # ATR for volatility normalization and trailing stop
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(tr1, np.abs(low[1:] - close[:-1]))
    tr = np.concatenate([[np.nan], tr2])
    atr = np.zeros(n)
    atr_period = 14
    for i in range(n):
        if i < atr_period:
            atr[i] = np.nan
        else:
            atr[i] = np.nanmean(tr[i-atr_period+1:i+1])
    
    # EMA trend filter on 6h
    ema_fast = np.zeros(n)
    ema_slow = np.zeros(n)
    fast_period = 9
    slow_period = 21
    ema_fast[:] = np.nan
    ema_slow[:] = np.nan
    
    # Calculate EMA using recursive formula
    if n > 0:
        ema_fast[fast_period-1] = np.mean(close[:fast_period])
        ema_slow[slow_period-1] = np.mean(close[:slow_period])
        for i in range(fast_period, n):
            ema_fast[i] = (close[i] * 2/(fast_period+1)) + ema_fast[i-1] * (1 - 2/(fast_period+1))
        for i in range(slow_period, n):
            ema_slow[i] = (close[i] * 2/(slow_period+1)) + ema_slow[i-1] * (1 - 2/(slow_period+1))
    
    # Volume moving average for confirmation
    vol_ma = np.zeros(n)
    vol_period = 20
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= vol_period:
            vol_sum -= volume[i-vol_period]
        if i < vol_period - 1:
            vol_ma[i] = np.nan
        else:
            vol_ma[i] = vol_sum / vol_period
    
    # Get 12h trend (EMA crossover)
    close_12h = df_12h['close'].values
    ema_12h_fast = np.zeros(len(close_12h))
    ema_12h_slow = np.zeros(len(close_12h))
    ema_12h_fast[:] = np.nan
    ema_12h_slow[:] = np.nan
    
    if len(close_12h) > 0:
        ema_12h_fast[8] = np.mean(close_12h[:9])  # 9-period
        ema_12h_slow[20] = np.mean(close_12h[:21])  # 21-period
        for i in range(9, len(close_12h)):
            ema_12h_fast[i] = (close_12h[i] * 2/10) + ema_12h_fast[i-1] * 0.8
        for i in range(21, len(close_12h)):
            ema_12h_slow[i] = (close_12h[i] * 2/22) + ema_12h_slow[i-1] * (10/11)
    
    # Align 12h EMA to 6h
    ema_12h_fast_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_fast)
    ema_12h_slow_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_slow)
    
    # Trend is up when fast EMA > slow EMA on 12h
    trend_up = ema_12h_fast_aligned > ema_12h_slow_aligned
    trend_down = ema_12h_fast_aligned < ema_12h_slow_aligned
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 21, 20)  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(atr[i]) or np.isnan(ema_fast[i]) or np.isnan(ema_slow[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(trend_up[i]) or np.isnan(trend_down[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Calculate adaptive position size based on volatility and trend strength
        # Base size adjusted by ATR (inverse volatility) and trend alignment
        atr_ratio = atr[i] / close[i] if close[i] > 0 else 0
        # Normalize ATR ratio to 0-1 range (assuming typical ATR% of 1-5%)
        vol_factor = np.clip(1.0 - (atr_ratio - 0.01) / 0.04, 0.5, 1.0)  # Lower vol = higher factor
        
        # Trend strength: distance between EMAs normalized by price
        if close[i] > 0:
            ema_dist = abs(ema_fast[i] - ema_slow[i]) / close[i]
            trend_strength = np.clip(ema_dist * 100, 0.5, 2.0)  # Scale to reasonable range
        else:
            trend_strength = 1.0
        
        # Volume confirmation: volume above average suggests participation
        vol_confirm = volume[i] > vol_ma[i] * 1.5  # 50% above average
        
        # Entry conditions
        if position == 0:
            # LONG: 6h EMA bullish crossover AND 12h trend up AND volume confirmation
            if ema_fast[i] > ema_slow[i] and ema_fast[i-1] <= ema_slow[i-1] and trend_up[i] and vol_confirm:
                # Adaptive size: base 0.25 adjusted by vol factor and trend strength
                size = 0.25 * vol_factor * min(trend_strength, 1.5)
                size = np.clip(size, 0.15, 0.35)
                signals[i] = size
                position = 1
            # SHORT: 6h EMA bearish crossover AND 12h trend down AND volume confirmation
            elif ema_fast[i] < ema_slow[i] and ema_fast[i-1] >= ema_slow[i-1] and trend_down[i] and vol_confirm:
                size = 0.25 * vol_factor * min(trend_strength, 1.5)
                size = np.clip(size, 0.15, 0.35)
                signals[i] = -size
                position = -1
        elif position == 1:
            # EXIT LONG: 6h EMA bearish crossover OR 12h trend fails OR volatility spike
            if ema_fast[i] < ema_slow[i] or not trend_up[i] or (atr[i] > atr[i-1] * 2 and vol_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 * vol_factor * min(trend_strength, 1.5)
                signals[i] = np.clip(signals[i], 0.15, 0.35)
        elif position == -1:
            # EXIT SHORT: 6h EMA bullish crossover OR 12h trend fails OR volatility spike
            if ema_fast[i] > ema_slow[i] or not trend_down[i] or (atr[i] > atr[i-1] * 2 and vol_confirm):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25 * vol_factor * min(trend_strength, 1.5)
                signals[i] = np.clip(signals[i], -0.35, -0.15)
    
    return signals