#!/usr/bin/env python3
name = "1d_Weekly_Keltner_Breakout_Trend"
timeframe = "1d"
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
    
    # Get weekly data for ATR and EMA
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Weekly ATR for Keltner channels
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    tr1 = high_1w - low_1w
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Weekly EMA for middle line
    ema = pd.Series(close_1w).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Keltner channels
    keltner_upper = ema + (2.0 * atr)
    keltner_lower = ema - (2.0 * atr)
    
    # Align to daily
    keltner_upper_aligned = align_htf_to_ltf(prices, df_1w, keltner_upper)
    keltner_lower_aligned = align_htf_to_ltf(prices, df_1w, keltner_lower)
    ema_aligned = align_htf_to_ltf(prices, df_1w, ema)
    
    # Daily trend filter: price above/below 50-day EMA
    ema_50 = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(keltner_upper_aligned[i]) or np.isnan(keltner_lower_aligned[i]) or
            np.isnan(ema_aligned[i]) or np.isnan(ema_50[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Keltner upper AND above daily EMA50
            if close[i] > keltner_upper_aligned[i] and close[i] > ema_50[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Keltner lower AND below daily EMA50
            elif close[i] < keltner_lower_aligned[i] and close[i] < ema_50[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price falls below Keltner middle (EMA)
            if close[i] < ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: price rises above Keltner middle (EMA)
            if close[i] > ema_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals