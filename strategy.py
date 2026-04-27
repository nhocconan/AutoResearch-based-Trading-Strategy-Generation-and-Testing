#!/usr/bin/env python3
"""
6h_Keltner_Breakout_12hTrend_VolumeFilter
Hypothesis: Use Keltner Channel breakouts (2x ATR) with 12h EMA50 trend filter and volume spike (>1.8x 20-period average). Keltner adapts to volatility, reducing false breakouts in low-volatility chop. Trend filter ensures trades align with higher timeframe momentum. Volume filter confirms institutional interest. Designed for 15-35 trades/year to minimize fee drift. Works in bull (breakouts with trend) and bear (mean reversion at extremes with trend filter).
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
    
    # Calculate Keltner Channel: EMA20 +/- 2*ATR(10)
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_raw = pd.Series(np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))).rolling(window=10, min_periods=10).mean().values
    kc_upper = ema_20 + 2.0 * atr_raw
    kc_lower = ema_20 - 2.0 * atr_raw
    
    # Calculate 12h EMA50 for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:
        return np.zeros(n)
    ema_50_12h = pd.Series(df_12h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation: current volume > 1.8 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.8 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for EMA20, ATR(10), EMA50, and volume average
    start_idx = max(20, 10, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(volume_confirm[i])):
            signals[i] = 0.0
            continue
        
        kc_upper_val = kc_upper[i]
        kc_lower_val = kc_lower[i]
        ema_50_val = ema_50_12h_aligned[i]
        vol_conf = volume_confirm[i]
        
        if position == 0:
            # Long: price breaks above Keltner upper with volume confirmation AND above 12h EMA50 (uptrend)
            if close[i] > kc_upper_val and vol_conf and close[i] > ema_50_val:
                signals[i] = size
                position = 1
            # Short: price breaks below Keltner lower with volume confirmation AND below 12h EMA50 (downtrend)
            elif close[i] < kc_lower_val and vol_conf and close[i] < ema_50_val:
                signals[i] = -size
                position = -1
        elif position == 1:
            # Exit long: price closes below Keltner lower (mean reversion)
            if close[i] < kc_lower_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price closes above Keltner upper (mean reversion)
            if close[i] > kc_upper_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Keltner_Breakout_12hTrend_VolumeFilter"
timeframe = "6h"
leverage = 1.0