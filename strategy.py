#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout_1dTrend_Volume
Hypothesis: Breakout from 4h Keltner Channel (2*ATR) in direction of 1d EMA34 trend, with volume confirmation.
Keltner adapts to volatility, reducing whipsaw in ranging markets. Works in both bull and bear regimes by following higher timeframe trend.
Target: 75-200 total trades over 4 years (19-50/year) to minimize fee drag.
"""

name = "4h_Keltner_Channel_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Daily EMA34 for trend filter
    ema_34 = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Get 4h price, volume, and ATR for Keltner
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR(10) for Keltner Channel
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First value
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Calculate EMA20 for Keltner middle line
    ema20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner Channel bands
    keltner_upper = ema20 + 2 * atr
    keltner_lower = ema20 - 2 * atr
    
    # Volume filter: current volume > 2.0x 20-period EMA
    vol_ema20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_filter = volume > vol_ema20 * 2.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA34 (34 bars), ATR/EMA20 (20), volume EMA (20)
    start_idx = 34
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_aligned[i]) or 
            np.isnan(keltner_upper[i]) or
            np.isnan(keltner_lower[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: above EMA34 (uptrend) AND price breaks above Keltner Upper with volume spike
            if close[i] > ema_34_aligned[i] and high[i] > keltner_upper[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: below EMA34 (downtrend) AND price breaks below Keltner Lower with volume spike
            elif close[i] < ema_34_aligned[i] and low[i] < keltner_lower[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price breaks below Keltner Middle OR trend turns bearish
            if low[i] < ema20[i] or close[i] < ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price breaks above Keltner Middle OR trend turns bullish
            if high[i] > ema20[i] or close[i] > ema_34_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals