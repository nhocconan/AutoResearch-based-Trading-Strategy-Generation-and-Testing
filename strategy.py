#!/usr/bin/env python3
"""
6h_OrderFlow_Imbalance_1dTrend_Filter
Hypothesis: Use 1d price action (higher highs/lows) to determine trend direction, and 6h order flow imbalance (delta volume) for entry timing. 
In bull markets: buy pullbacks in uptrend when buying pressure exceeds selling pressure.
In bear markets: sell rallies in downtrend when selling pressure exceeds buying pressure.
Volume confirmation reduces false signals. Designed to work in both trending and ranging markets by requiring trend alignment.
"""

name = "6h_OrderFlow_Imbalance_1dTrend_Filter"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    taker_buy_volume = prices['taker_buy_volume'].values
    
    # Calculate selling volume (market sells)
    sell_volume = volume - taker_buy_volume
    # Calculate volume delta (buying pressure - selling pressure)
    volume_delta = taker_buy_volume - sell_volume
    
    # Smooth volume delta with 3-period EMA to reduce noise
    delta_ema = pd.Series(volume_delta).ewm(span=3, adjust=False, min_periods=3).mean().values
    
    # Get 1d data for trend direction
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Determine 1d trend: higher highs and higher lows = uptrend, lower highs and lower lows = downtrend
    # Use 2-period lookback for swing points
    hh = high_1d >= np.roll(high_1d, 1)  # Higher high
    hl = low_1d >= np.roll(low_1d, 1)    # Higher low
    lh = high_1d <= np.roll(high_1d, 1)  # Lower high
    ll = low_1d <= np.roll(low_1d, 1)    # Lower low
    
    # Trend direction: 1 for uptrend (HH & HL), -1 for downtrend (LH & LL), 0 for mixed/unclear
    trend_1d = np.where(hh & hl, 1, np.where(lh & ll, -1, 0))
    
    # Align 1d trend to 6h timeframe
    trend_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_1d.astype(float))
    
    # Volume filter: current volume > 1.3x 20-period average to ensure participation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(trend_1d_aligned[i]) or 
            np.isnan(delta_ema[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: 1d uptrend AND buying pressure (delta > 0) AND volume filter
            if trend_1d_aligned[i] == 1 and delta_ema[i] > 0 and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: 1d downtrend AND selling pressure (delta < 0) AND volume filter
            elif trend_1d_aligned[i] == -1 and delta_ema[i] < 0 and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: 1d trend turns down OR selling pressure emerges (delta < 0)
            if trend_1d_aligned[i] == -1 or delta_ema[i] < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: 1d trend turns up OR buying pressure emerges (delta > 0)
            if trend_1d_aligned[i] == 1 or delta_ema[i] > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals