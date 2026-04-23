#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla H4/L4 breakout with 1d EMA50 trend filter and volume spike.
- Camarilla levels (H4, L4) from prior 1d: major support/resistance (stronger than R3/S3)
- Long: price breaks above H4 + volume > 2.0x 20-period avg + price > 1d EMA50
- Short: price breaks below L4 + volume > 2.0x 20-period avg + price < 1d EMA50
- Exit: price re-enters Camarilla H3-L3 range OR EMA50 trend flip
- Uses 12h primary timeframe for lower frequency, reducing fee drag
- Discrete position sizing: ±0.25 to minimize fee churn
- Works in bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
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
    
    # Volume confirmation: > 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1d Camarilla levels (based on prior 1d OHLC)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values
    
    # Camarilla formula: range = high - low
    # H4 = close + (high - low) * 1.1/2
    # L4 = close - (high - low) * 1.1/2
    # H3 = close + (high - low) * 1.1/6
    # L3 = close - (high - low) * 1.1/6
    rng = high_1d - low_1d
    camarilla_h4 = close_1d_prev + rng * (1.1 / 2)
    camarilla_l4 = close_1d_prev - rng * (1.1 / 2)
    camarilla_h3 = close_1d_prev + rng * (1.1 / 6)
    camarilla_l3 = close_1d_prev - rng * (1.1 / 6)
    
    # Align Camarilla levels to 12h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # Need 50 for EMA50, 20 for volume MA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(h4_aligned[i]) or
            np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above H4 + volume confirmation + price > 1d EMA50
            if (close[i] > h4_aligned[i] and 
                volume_confirm and 
                close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L4 + volume confirmation + price < 1d EMA50
            elif (close[i] < l4_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price re-enters below H3 (mean reversion) OR price < 1d EMA50 (trend flip)
            if close[i] < h3_aligned[i] or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price re-enters above L3 (mean reversion) OR price > 1d EMA50 (trend flip)
            if close[i] > l3_aligned[i] or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_H4L4_Breakout_1dEMA50_VolumeSpike"
timeframe = "12h"
leverage = 1.0