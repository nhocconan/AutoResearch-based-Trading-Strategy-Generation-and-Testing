#!/usr/bin/env python3
# 6h_12h_keltner_reversion_v1
# Strategy: Mean reversion at Keltner bands with 12h trend filter
# Timeframe: 6h
# Leverage: 1.0
# Hypothesis: Price reverts to EMA when touching Keltner bands (ATR-based) during strong 12h trends.
# Works in bull by buying dips in uptrend, in bear by selling rallies in downtrend.
# Uses ATR(20) for band width and EMA(20) for mean. Trend filter: EMA(50) slope on 12h.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_12h_keltner_reversion_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Price arrays
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12h EMA(50) for trend filter
    close_12h = df_12h['close'].values
    ema_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periodos=50).mean().values
    ema_12h_slope = np.diff(ema_12h, prepend=ema_12h[0])
    ema_12h_up = ema_12h_slope > 0  # Uptrend when rising
    ema_12h_up_aligned = align_htf_to_ltf(prices, df_12h, ema_12h_up)
    
    # 6h EMA(20) - mean
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # 6h ATR(20) for Keltner bands
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = tr2[0] if len(tr2) > 0 else 0
    tr3[0] = tr3[0] if len(tr3) > 0 else 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_20 = pd.Series(tr).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner bands: EMA(20) ± 2.0 * ATR(20)
    keltner_upper = ema_20 + 2.0 * atr_20
    keltner_lower = ema_20 - 2.0 * atr_20
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(ema_20[i]) or np.isnan(keltner_upper[i]) or np.isnan(keltner_lower[i]) or
            np.isnan(ema_12h_up_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Long: price touches lower band in 12h uptrend
        if low[i] <= keltner_lower[i] and ema_12h_up_aligned[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: price touches upper band in 12h downtrend
        elif high[i] >= keltner_upper[i] and not ema_12h_up_aligned[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: price crosses EMA(20) or trend change
        elif position == 1 and (close[i] >= ema_20[i] or not ema_12h_up_aligned[i]):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] <= ema_20[i] or ema_12h_up_aligned[i]):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals