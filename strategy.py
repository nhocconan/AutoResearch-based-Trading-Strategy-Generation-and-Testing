#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with 1w Camarilla H3/L3 filter and volume confirmation.
- Primary timeframe: 6h for entries/exits.
- HTF: 1w Camarilla H3 (resistance) and L3 (support) levels for institutional reference points.
  Breakouts above H3 with volume = strong bullish continuation.
  Breakdowns below L3 with volume = strong bearish continuation.
- Volume: Current 6h volume > 1.5 * 20-period 6h volume MA to confirm breakout strength.
- Entry: Long when price breaks above Donchian(20) high AND price > 1w H3 AND volume spike.
         Short when price breaks below Donchian(20) low AND price < 1w L3 AND volume spike.
- Exit: Opposite Donchian break (close below 20-period low for long, above 20-period high for short).
- Signal size: 0.25 discrete to limit drawdown and reduce fee churn.
- Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe.
Donchian breakouts capture momentum, Camarilla H3/L3 acts as institutional break/continuation filter,
volume confirmation reduces false signals. Works in trending markets (both bull and bear).
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
    
    # Calculate Donchian(20) channels on 6h
    period20_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    period20_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period volume MA on 6h for volume confirmation
    vol_ma_6h = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1w data for Camarilla levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 1:
        return np.zeros(n)
    
    # Calculate Camarilla levels on weekly data
    # Camarilla: H4 = close + 1.5*(high-low), H3 = close + 1.1*(high-low)
    # L3 = close - 1.1*(high-low), L4 = close - 1.5*(high-low)
    df_1w_high = df_1w['high'].values
    df_1w_low = df_1w['low'].values
    df_1w_close = df_1w['close'].values
    
    camarilla_range = df_1w_high - df_1w_low
    h3_level = df_1w_close + 1.1 * camarilla_range
    l3_level = df_1w_close - 1.1 * camarilla_range
    
    # Align HTF indicators to 6h
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3_level)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3_level)
    
    # Volume confirmation: current 6h volume > 1.5 * 20-period 6h volume MA
    volume_spike = volume > (1.5 * vol_ma_6h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = 20  # Need enough bars for Donchian
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(period20_high[i]) or np.isnan(period20_low[i]) or 
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        
        if position == 0:
            # Check for entry signals with volume spike
            if volume_spike[i]:
                # Bullish breakout: price breaks above Donchian high AND above 1w H3
                if curr_high > period20_high[i] and curr_close > h3_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                # Bearish breakdown: price breaks below Donchian low AND below 1w L3
                elif curr_low < period20_low[i] and curr_close < l3_aligned[i]:
                    signals[i] = -0.25
                    position = -1
        elif position == 1:
            # Long exit: price breaks below Donchian low (20-period low)
            if curr_low < period20_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Donchian high (20-period high)
            if curr_high > period20_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_1wCamarilla_H3L3_VolumeSpike_v1"
timeframe = "6h"
leverage = 1.0