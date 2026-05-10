#!/usr/bin/env python3
# 6h_12h_1d_Keltner_Breakout_Trend_Volume
# Hypothesis: 6h Keltner Channel breakout with 12h trend filter (EMA50) and volume confirmation.
# Uses Keltner Channel (EMA-based) instead of Donchian for adaptive volatility bands.
# Works in bull/bear by requiring trend alignment, avoiding counter-trend traps.
# Targets 50-150 total trades over 4 years via strict multi-condition entry.

name = "6h_12h_1d_Keltner_Breakout_Trend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get 12h data for trend filter and Keltner calculation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # 6h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 12h EMA20 for Keltner middle line
    ema_20_12h = pd.Series(df_12h['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Calculate 12h ATR(10) for Keltner width
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = high_12h[0] - low_12h[0]  # first bar TR
    tr2[0] = np.abs(high_12h[0] - close_12h[0])
    tr3[0] = np.abs(low_12h[0] - close_12h[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_10_12h = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel: EMA20 ± 2*ATR(10)
    keltner_upper = ema_20_12h + 2 * atr_10_12h
    keltner_lower = ema_20_12h - 2 * atr_10_12h
    
    # Align Keltner levels to 6h timeframe (wait for 12h bar to close)
    kc_upper_aligned = align_htf_to_ltf(prices, df_12h, keltner_upper)
    kc_lower_aligned = align_htf_to_ltf(prices, df_12h, keltner_lower)
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume average (4-period for 6h = 1 day)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need enough history for Keltner (20 periods) + EMA50 + vol MA
    start_idx = 60
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(kc_upper_aligned[i]) or
            np.isnan(kc_lower_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend: 12h close > EMA50
        close_12h_aligned = align_htf_to_ltf(prices, df_12h, df_12h['close'].values)
        uptrend = close_12h_aligned[i] > ema_50_12h_aligned[i]
        downtrend = close_12h_aligned[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: Breakout above Keltner upper in uptrend with volume
            if close[i] > kc_upper_aligned[i] and uptrend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: Breakdown below Keltner lower in downtrend with volume
            elif close[i] < kc_lower_aligned[i] and downtrend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: close below Keltner upper or trend fails
                if close[i] < kc_upper_aligned[i] or not uptrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: close above Keltner lower or trend fails
                if close[i] > kc_lower_aligned[i] or not downtrend:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals