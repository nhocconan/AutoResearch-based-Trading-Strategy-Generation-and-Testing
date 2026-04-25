#!/usr/bin/env python3
"""
6h_RegimeAdaptive_Camarilla_Volume
Hypothesis: On 6h timeframe, use Camarilla pivot levels (H3/L3 for mean reversion in ranging markets, H4/L4 for breakout continuation in trending markets) with regime filter based on Bollinger Bandwidth percentile. In low volatility regimes (BBWP < 30%), fade at H3/L3. In high volatility regimes (BBWP > 70%), breakout at H4/L4. Volume confirmation (>1.5x 20-bar average) required for all entries. Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag. Works in both bull and bear markets by adapting to volatility regimes.
"""

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
    
    # 1d data for Camarilla pivots and Bollinger Bandwidth (loaded ONCE)
    df_1d = get_htf_data(prices, '1d')
    
    # 1d Camarilla pivot levels (based on previous day's OHLC)
    # H4 = close + 1.5*(high-low), L4 = close - 1.5*(high-low)
    # H3 = close + 1.125*(high-low), L3 = close - 1.125*(high-low)
    # We use the previous day's values to avoid look-ahead
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    prev_range = prev_high - prev_low
    
    H4 = prev_close + 1.5 * prev_range
    L4 = prev_close - 1.5 * prev_range
    H3 = prev_close + 1.125 * prev_range
    L3 = prev_close - 1.125 * prev_range
    
    # Align 1d pivot levels to 6h timeframe (waits for completed 1d bar)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    
    # 1d Bollinger Bandwidth for regime filter (20, 2)
    bb_middle = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).mean().values
    bb_std = pd.Series(df_1d['close'].values).rolling(window=20, min_periods=20).std().values
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    bb_width = bb_upper - bb_lower
    bb_width_percentile = pd.Series(bb_width).rolling(window=50, min_periods=10).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    bb_width_percentile_aligned = align_htf_to_ltf(prices, df_1d, bb_width_percentile)
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    # Start index: need enough for 1d indicators (50 for BBWP percentile, 20 for vol MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        bars_since_entry += 1
        
        # Skip if any data not ready
        if (np.isnan(H4_aligned[i]) or np.isnan(L4_aligned[i]) or np.isnan(H3_aligned[i]) or 
            np.isnan(L3_aligned[i]) or np.isnan(bb_width_percentile_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Regime determination based on Bollinger Bandwidth percentile
        bbwp = bb_width_percentile_aligned[i]
        low_vol_regime = bbwp < 30   # Ranging market: mean revert at H3/L3
        high_vol_regime = bbwp > 70  # Trending market: breakout at H4/L4
        
        if position == 0:
            # Look for entry signals
            # Long signals
            long_mean_revert = (curr_close <= L3_aligned[i]) and low_vol_regime
            long_breakout = (curr_close >= H4_aligned[i]) and high_vol_regime
            
            # Short signals
            short_mean_revert = (curr_close >= H3_aligned[i]) and low_vol_regime
            short_breakout = (curr_close <= L4_aligned[i]) and high_vol_regime
            
            long_entry = (long_mean_revert or long_breakout) and volume_spike[i]
            short_entry = (short_mean_revert or short_breakout) and volume_spike[i]
            
            if long_entry:
                signals[i] = 0.25
                position = 1
                entry_price = curr_close
                bars_since_entry = 0
            elif short_entry:
                signals[i] = -0.25
                position = -1
                entry_price = curr_close
                bars_since_entry = 0
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit when price reverts to mean (H3 in low vol) or breaks down (L4 in high vol)
            # Or when regime changes against position
            if low_vol_regime and curr_close >= H3_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif high_vol_regime and curr_close <= L4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short position: exit when price reverts to mean (L3 in low vol) or breaks up (H4 in high vol)
            # Or when regime changes against position
            if low_vol_regime and curr_close <= L3_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif high_vol_regime and curr_close >= H4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_RegimeAdaptive_Camarilla_Volume"
timeframe = "6h"
leverage = 1.0