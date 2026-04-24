#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla H3/L3 breakout with 1d EMA34 trend filter and volume confirmation.
- Primary timeframe: 6h for execution, HTF: 1d for EMA34 trend direction.
- EMA34 > rising: bullish trend bias; EMA34 < falling: bearish trend bias.
- Entry: Long when price breaks above Camarilla H3 AND EMA34 trending up.
         Short when price breaks below Camarilla L3 AND EMA34 trending down.
         In weak trend (EMA34 flat): fade at H3/L3 with reversal confirmation.
- Exit: Opposite Camarilla breakout or EMA34 trend reversal.
- Volume confirmation: current volume > 1.5 * 20-period volume MA (to avoid false breakouts).
- Discrete signal size: 0.25 to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
- Novelty: Combines Camarilla pivot breakout logic with 1d EMA trend regime (not recently tried on 6h).
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
    
    # Get 1d data for Camarilla pivot and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 40:
        return np.zeros(n)
    
    # Calculate 1d Camarilla pivot levels (based on previous day)
    # Camarilla uses yesterday's high, low, close
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    pivot = (prev_high + prev_low + prev_close) / 3.0
    range_hl = prev_high - prev_low
    
    # Camarilla levels
    H3 = pivot + (range_hl * 1.1 / 4)
    L3 = pivot - (range_hl * 1.1 / 4)
    H4 = pivot + (range_hl * 1.1 / 2)
    L4 = pivot - (range_hl * 1.1 / 2)
    
    # Calculate 1d EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    # EMA slope: positive = rising, negative = falling
    ema_slope = np.diff(ema_34, prepend=ema_34[0])
    
    # Align 1d indicators to 6h
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    ema_slope_aligned = align_htf_to_ltf(prices, df_1d, ema_slope)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA (on 6h)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(40, 34, 20)  # Need enough 1d bars for EMA34 and Camarilla
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(ema_slope_aligned[i]) or 
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        prev_close = close[i-1]
        h3 = H3_aligned[i]
        l3 = L3_aligned[i]
        h4 = H4_aligned[i]
        l4 = L4_aligned[i]
        ema = ema_34_aligned[i]
        ema_slope_val = ema_slope_aligned[i]
        
        if position == 0:
            # Check for entry signals
            if volume_spike[i]:
                # Strong trend: EMA slope significant
                if abs(ema_slope_val) > 0.1:  # Trending regime: breakout strategy
                    if ema_slope_val > 0:  # Bullish trend
                        if curr_close > h3:  # Break above H3
                            signals[i] = 0.25
                            position = 1
                    else:  # Bearish trend
                        if curr_close < l3:  # Break below L3
                            signals[i] = -0.25
                            position = -1
                else:  # Weak trend (EMA flat): mean reversion at extremes
                    # Fade at H3/L3 with reversal confirmation
                    if curr_high >= h3 and curr_close < curr_high:  # Rejection at H3
                        signals[i] = -0.25
                        position = -1
                    elif curr_low <= l3 and curr_close > curr_low:  # Rejection at L3
                        signals[i] = 0.25
                        position = 1
        elif position == 1:
            # Long exit: price breaks below L3 OR EMA slope turns bearish
            if curr_close < l3 or ema_slope_val < -0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above H3 OR EMA slope turns bullish
            if curr_close > h3 or ema_slope_val > 0.05:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_1dEMA34Trend_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0