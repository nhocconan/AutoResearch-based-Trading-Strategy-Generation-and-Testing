#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly Camarilla H4/L4 filter and volume confirmation.
- Uses Donchian channel breakouts (20-bar high/low) on 6h timeframe to capture medium-term momentum.
- Weekly Camarilla H4/L4 levels act as strong support/resistance: only allow long when price > weekly H4,
  short when price < weekly L4 to avoid counter-trend trades.
- Volume confirmation: current volume > 1.5x 20-bar average to filter weak breakouts.
- Designed for 6h timeframe to work in both bull and bear markets by aligning with weekly structure.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Prior completed weekly OHLC for Camarilla H4/L4
    high_1w = df_1w['high'].shift(1).values
    low_1w = df_1w['low'].shift(1).values
    close_1w = df_1w['close'].shift(1).values
    
    # Weekly Camarilla levels: H4 = close + 1.5*(high-low)/2, L4 = close - 1.5*(high-low)/2
    camarilla_h4 = close_1w + 1.5 * (high_1w - low_1w) / 2
    camarilla_l4 = close_1w - 1.5 * (high_1w - low_1w) / 2
    
    # Align weekly Camarilla to 6h timeframe
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1w, camarilla_l4)
    
    # Donchian channel (20-bar) on 6h
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(60, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.5x average)
        volume_confirm = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above Donchian high AND price > weekly H4 AND volume confirmation
            if close[i] > donchian_high[i] and close[i] > camarilla_h4_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND price < weekly L4 AND volume confirmation
            elif close[i] < donchian_low[i] and close[i] < camarilla_l4_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below Donchian low OR price < weekly H4
            if close[i] < donchian_low[i] or close[i] < camarilla_h4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above Donchian high OR price > weekly L4
            if close[i] > donchian_high[i] or close[i] > camarilla_l4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyCamarilla_H4L4_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0