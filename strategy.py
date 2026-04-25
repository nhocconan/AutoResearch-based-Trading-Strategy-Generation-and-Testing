#!/usr/bin/env python3
"""
6h Camarilla H3/L3 Breakout with 12h EMA50 Trend and Volume Spike
Hypothesis: Camarilla H3/L3 levels act as key intraday resistance/support on 12h timeframe.
Breakouts above H3 or below L3 with volume confirmation and aligned with 12h EMA50 trend
capture momentum in both bull (breakouts above H3 in uptrend) and bear (breakouts below L3 in downtrend) markets.
Uses discrete position sizing (0.0, ±0.25) to minimize fee churn.
Designed for 6h timeframe with tight entry conditions to achieve 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for Camarilla pivot and EMA (call ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Camarilla pivot levels on 12h
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Camarilla calculations
    rng = high_12h - low_12h
    H3 = close_12h + rng * 1.1 / 4
    L3 = close_12h - rng * 1.1 / 4
    H4 = close_12h + rng * 1.1 / 2
    L4 = close_12h - rng * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe
    H3_aligned = align_htf_to_ltf(prices, df_12h, H3)
    L3_aligned = align_htf_to_ltf(prices, df_12h, L3)
    H4_aligned = align_htf_to_ltf(prices, df_12h, H4)
    L4_aligned = align_htf_to_ltf(prices, df_12h, L4)
    
    # Calculate EMA50 on 12h close for trend
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Calculate volume spike: current volume > 2.0 * 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA and volume MA
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        ema_trend = ema_50_12h_aligned[i]
        vol_spike = volume_spike[i]
        H3_level = H3_aligned[i]
        L3_level = L3_aligned[i]
        H4_level = H4_aligned[i]
        L4_level = L4_aligned[i]
        
        if position == 0:
            # Look for entry signals
            # Long: price breaks above H3 (resistance) AND volume spike AND price > EMA (uptrend)
            # But only if not already above H4 (avoid chasing extreme breakouts)
            long_entry = (curr_high > H3_level) and vol_spike and (curr_close > ema_trend) and (curr_close < H4_level)
            # Short: price breaks below L3 (support) AND volume spike AND price < EMA (downtrend)
            # But only if not already below L4 (avoid chasing extreme breakouts)
            short_entry = (curr_low < L3_level) and vol_spike and (curr_close < ema_trend) and (curr_close > L4_level)
            
            if long_entry:
                signals[i] = 0.25
                position = 1
            elif short_entry:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position management
            # Exit: price crosses below H3 OR price crosses below EMA (trend change)
            if (curr_low < H3_level) or (curr_close < ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position management
            # Exit: price crosses above L3 OR price crosses above EMA (trend change)
            if (curr_high > L3_level) or (curr_close > ema_trend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H3L3_Breakout_12hEMA50_Trend_VolumeSpike"
timeframe = "6h"
leverage = 1.0