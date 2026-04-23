#!/usr/bin/env python3
"""
Hypothesis: 6h Camarilla Pivot Breakout with 12h EMA trend filter and volume spike confirmation.
- Uses actual Binance 12h data for EMA50 trend (avoid counter-trend trades)
- Camarilla pivot levels calculated from prior 1d bar (H3/L3 for mean reversion, H4/L4 for breakout)
- Long when price breaks above H4 with volume > 2x average AND 12h EMA50 rising
- Short when price breaks below L4 with volume > 2x average AND 12h EMA50 falling
- Position size: 0.25 discrete level
- Target: 12-37 trades/year on 6h timeframe (50-150 total over 4 years)
- Works in both bull/bear via 12h trend filter and volatility-adjusted breakouts
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
    
    # Volume confirmation: > 2.0x 24-period average (6h * 4 = 1 day)
    vol_ma = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    # 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Previous 1d high/low/close for Camarilla pivots (using 1d data)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels from previous 1d bar
    # H4/L4 = breakout levels, H3/L3 = mean reversion levels
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h4 = close_1d + range_1d * 1.1 / 2.0
    l4 = close_1d - range_1d * 1.1 / 2.0
    h3 = close_1d + range_1d * 1.1 / 4.0
    l3 = close_1d - range_1d * 1.1 / 4.0
    
    # Align 1d Camarilla levels to 6h timeframe
    h4_aligned = align_htf_to_ltf(prices, df_1d, h4)
    l4_aligned = align_htf_to_ltf(prices, df_1d, l4)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(24, 50)  # volume MA, 12h EMA
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma[i]) or np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(h4_aligned[i]) or np.isnan(l4_aligned[i]) or
            np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation (> 2.0x average)
        volume_confirm = volume[i] > 2.0 * vol_ma[i]
        
        # 12h EMA trend: rising/falling
        if i >= 1:
            ema_rising = ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]
            ema_falling = ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]
        else:
            ema_rising = False
            ema_falling = False
        
        if position == 0:
            # Long breakout: price > H4 AND volume confirmation AND 12h EMA rising
            if close[i] > h4_aligned[i] and volume_confirm and ema_rising:
                signals[i] = 0.25
                position = 1
            # Short breakout: price < L4 AND volume confirmation AND 12h EMA falling
            elif close[i] < l4_aligned[i] and volume_confirm and ema_falling:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price < H3 (mean reversion) OR 12h EMA turns flat/falling
            if close[i] < h3_aligned[i] or not ema_rising:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price > L3 (mean reversion) OR 12h EMA turns flat/rising
            if close[i] > l3_aligned[i] or not ema_falling:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_H4L4_Breakout_12hEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0