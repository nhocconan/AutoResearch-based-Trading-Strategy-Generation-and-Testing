#!/usr/bin/env python3
# 4h_Keltner_Channel_Momentum_v1
# Hypothesis: Keltner Channel (ATR-based) captures volatility-adjusted breakouts.
# In bull markets, price breaks above upper KC in uptrend (EMA20 > EMA50) signals momentum continuation.
# In bear markets, price breaks below lower KC in downtrend (EMA20 < EMA50) signals momentum continuation.
# Volume confirmation filters false breakouts. Uses daily EMA50 for trend filter to reduce whipsaw.
# Designed for low trade frequency (<50/year) to minimize fee drag.

name = "4h_Keltner_Channel_Momentum_v1"
timeframe = "4h"
leverage = 1.0

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
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate daily EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate EMA20 and ATR(10) for Keltner Channel on 4h
    ema_20 = pd.Series(close).ewm(span=20, adjust=False, min_periods=20).mean().values
    atr_10 = pd.Series(high - low).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # Keltner Channel: EMA20 ± 2 * ATR(10)
    kc_upper = ema_20 + 2 * atr_10
    kc_lower = ema_20 - 2 * atr_10
    
    # Volume confirmation (20-period MA)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need EMA20 (20), ATR(10) (10), volume MA (20), daily EMA50 (50)
    start_idx = max(20, 10, 20, 50)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_20[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or
            np.isnan(volume_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter: EMA20 vs EMA50 (daily)
        uptrend = ema_20[i] > ema_50_1d_aligned[i]
        downtrend = ema_20[i] < ema_50_1d_aligned[i]
        
        # Volume confirmation (>1.5x average to reduce noise)
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        if position == 0:
            # Long entry: uptrend + price breaks above KC upper + volume
            if uptrend and close[i] > kc_upper[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: downtrend + price breaks below KC lower + volume
            elif downtrend and close[i] < kc_lower[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend breaks or price re-enters below KC upper
            if not uptrend or close[i] < kc_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend breaks or price re-enters above KC lower
            if not downtrend or close[i] > kc_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals