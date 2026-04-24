#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot direction filter and volume confirmation.
- Uses Donchian channel (20-period high/low) from prior completed 6h candles for breakout signals.
- Weekly Camarilla pivot levels (H3/L3) from prior completed 1w candles to determine higher timeframe bias: 
  price above weekly H3 = bullish bias (only long), price below weekly L3 = bearish bias (only short).
- Volume confirmation: current volume > 2.0x 20-bar average to filter weak breakouts.
- Designed for 6h timeframe to capture medium-term breakouts aligned with weekly structure.
- Uses discrete position size 0.25 to limit drawdown and reduce fee churn.
- Targets 12-37 trades/year (50-150 total over 4 years) to stay fee-efficient.
- Weekly pivot filter adds structural bias to avoid counter-trend breakouts in ranging markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 6h data ONCE before loop for Donchian calculation
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 20:
        return np.zeros(n)
    
    # Get 1w data ONCE before loop for weekly Camarilla pivot filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Calculate Donchian channel (20-period) from 6h data
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    
    # Donchian high and low (20-period lookback)
    donchian_high = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 6h timeframe (wait for 6h bar to close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_6h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_6h, donchian_low)
    
    # Calculate weekly Camarilla pivot levels (H3, L3) from prior completed 1w candles
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Typical price for pivot calculation
    typical_price_1w = (high_1w + low_1w + close_1w) / 3.0
    # Camarilla width
    camarilla_width = (high_1w - low_1w) * 1.1 / 12.0
    # H3 and L3 levels (Camarilla: H3 = close + width*1.1, L3 = close - width*1.1)
    h3_1w = close_1w + camarilla_width * 1.1
    l3_1w = close_1w - camarilla_width * 1.1
    
    # Align weekly H3 and L3 to 6h timeframe (wait for 1w bar to close)
    h3_1w_aligned = align_htf_to_ltf(prices, df_1w, h3_1w)
    l3_1w_aligned = align_htf_to_ltf(prices, df_1w, l3_1w)
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(h3_1w_aligned[i]) or np.isnan(l3_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above Donchian high AND price above weekly H3 AND volume confirmation
            if close[i] > donchian_high_aligned[i] and close[i] > h3_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: breakout below Donchian low AND price below weekly L3 AND volume confirmation
            elif close[i] < donchian_low_aligned[i] and close[i] < l3_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: close below Donchian low OR price below weekly L3
            if close[i] < donchian_low_aligned[i] or close[i] < l3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: close above Donchian high OR price above weekly H3
            if close[i] > donchian_high_aligned[i] or close[i] > h3_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian20_Breakout_WeeklyCamarilla_H3L3_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0