#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla pivot breakout with 1d volume confirmation and 1w trend filter.
- Primary timeframe: 6h for execution, HTF: 1d for volume spike, 1w for trend direction.
- Camarilla levels calculated from prior 1d OHLC: H3/L3 for fade, H4/L4 for breakout.
- In 1d uptrend (price > 1w EMA50): Long breakout at H4 with volume spike.
- In 1d downtrend (price < 1w EMA50): Short breakdown at L4 with volume spike.
- Volume confirmation: current 6h volume > 1.5 * 20-period 6h volume MA.
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
    
    # Get 1d data for Camarilla pivots
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels from prior 1d bar
    # H4/L4 = breakout levels, H3/L3 = fade levels
    prior_high = df_1d['high'].shift(1).values
    prior_low = df_1d['low'].shift(1).values
    prior_close = df_1d['close'].shift(1).values
    prior_range = prior_high - prior_low
    
    # Camarilla calculations
    h4 = prior_close + prior_range * 1.1 / 2
    l4 = prior_close - prior_range * 1.1 / 2
    h3 = prior_close + prior_range * 1.1 / 4
    l3 = prior_close - prior_range * 1.1 / 4
    
    # Align 1d Camarilla levels to 6h
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA50 for trend direction
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 1w EMA50 and volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        # Trend filter: 1d trend based on 1w EMA50
        is_uptrend = ema_50_1w_aligned[i] < close[i]  # price above 1w EMA50 = uptrend
        is_downtrend = ema_50_1w_aligned[i] > close[i]  # price below 1w EMA50 = downtrend
        
        if position == 0:
            # Check for entry signals with volume confirmation
            if volume_spike[i]:
                # Uptrend: look for long breakout at H4
                if is_uptrend and curr_high > h4_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Downtrend: look for short breakdown at L4
                elif is_downtrend and curr_low < l4_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                # Optional: mean reversion at H3/L3 in ranging markets
                elif not is_uptrend and not is_downtrend:  # ranging
                    if curr_low <= l3_aligned[i] and curr_close > low[i]:
                        signals[i] = 0.25
                        position = 1
                    elif curr_high >= h3_aligned[i] and curr_close < high[i]:
                        signals[i] = -0.25
                        position = -1
        elif position == 1:
            # Long exit: price closes below H3 (fade level) or breakdown below L4
            if curr_close < h3_aligned[i] or curr_low < l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price closes above L3 (fade level) or breakout above H4
            if curr_close > l3_aligned[i] or curr_high > h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4L4_1dVolumeSpike_1wEMA50Trend_v1"
timeframe = "6h"
leverage = 1.0