#!/usr/bin/env python3
"""
Hypothesis: 1h Camarilla H3/L3 breakout with 4h EMA20 trend filter and volume spike.
- Primary timeframe: 1h (entry timing), HTF: 4h (trend filter) and 1d (Camarilla levels)
- Long: price breaks above 1d H3 + volume > 1.8x 24-period avg + price > 4h EMA20
- Short: price breaks below 1d L3 + volume > 1.8x 24-period avg + price < 4h EMA20
- Exit: price re-enters 1d H3-L3 range OR 4h EMA20 trend flip
- Session filter: 08-20 UTC to reduce noise trades
- Target: 60-150 total trades over 4 years (15-37/year) on 1h timeframe
- Discrete position sizing: ±0.20 to minimize fee churn
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
    
    # Pre-compute session hours (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # Volume confirmation: > 1.8x 24-period average (1h bars = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # Calculate 4h EMA20 for trend filter
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Calculate 1d Camarilla levels (based on prior 1d OHLC)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_prev = df_1d['close'].values  # Prior 1d close as reference
    
    # Camarilla formula: range = high - low
    # H3 = close + (high - low) * 1.1/6
    # L3 = close - (high - low) * 1.1/6
    rng = high_1d - low_1d
    camarilla_h3 = close_1d_prev + rng * (1.1 / 6)
    camarilla_l3 = close_1d_prev - rng * (1.1 / 6)
    
    # Align Camarilla levels to 1h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 20)  # Need 24 for volume MA, 20 for EMA20
    
    for i in range(start_idx, n):
        # Session filter: 08-20 UTC
        if not (8 <= hours[i] <= 20):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or 
            np.isnan(ema_20_4h_aligned[i]) or
            np.isnan(h3_aligned[i]) or
            np.isnan(l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 1.8x average)
        volume_confirm = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above H3 + volume confirmation + price > 4h EMA20
            if (close[i] > h3_aligned[i] and 
                volume_confirm and 
                close[i] > ema_20_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below L3 + volume confirmation + price < 4h EMA20
            elif (close[i] < l3_aligned[i] and 
                  volume_confirm and 
                  close[i] < ema_20_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price re-enters below L3 (mean reversion) OR price < 4h EMA20 (trend flip)
            if close[i] < l3_aligned[i] or close[i] < ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price re-enters above H3 (mean reversion) OR price > 4h EMA20 (trend flip)
            if close[i] > h3_aligned[i] or close[i] > ema_20_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA20_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0