#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1w EMA50 trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1w for EMA50 trend direction.
- EMA50 > rising: bullish bias, only take longs at H3 breakout or shorts at L3 breakdown.
- EMA50 < falling: bearish bias, only take shorts at L3 breakdown or longs at H3 breakout.
- Entry: Long when price closes above Camarilla H3 AND 1w EMA50 rising.
         Short when price closes below Camarilla L3 AND 1w EMA50 falling.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Exit: Opposite Camarilla level (L3 for long, H3 for short) or EMA50 trend flip.
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
    
    # Get 1w data for EMA50
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate EMA50 on 1w
    ema50 = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 1w EMA50 to 6h
    ema50_aligned = align_htf_to_ltf(prices, df_1w, ema50)
    
    # Calculate Camarilla levels from 1d (more stable than 6h)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Camarilla levels: based on previous day's range
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    range_ = prev_high - prev_low
    
    # Camarilla H3 and L3
    h3 = prev_close + range_ * 1.1 / 4
    l3 = prev_close - range_ * 1.1 / 4
    
    # Align 1d Camarilla levels to 6h
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need enough 1w bars for EMA50 and 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema50_aligned[i]) or np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        ema50_val = ema50_aligned[i]
        ema50_prev = ema50_aligned[i-1] if i > 0 else ema50_val
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        
        # EMA50 trend: rising if current > previous
        ema50_rising = ema50_val > ema50_prev
        ema50_falling = ema50_val < ema50_prev
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Bullish breakout: price closes above H3 AND EMA50 rising
                if curr_close > h3_aligned[i] and ema50_rising:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakdown: price closes below L3 AND EMA50 falling
                elif curr_close < l3_aligned[i] and ema50_falling:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price closes below L3 OR EMA50 starts falling
            if curr_close < l3_aligned[i] or ema50_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above H3 OR EMA50 starts rising
            if curr_close > h3_aligned[i] or ema50_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_1wEMA50_Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0