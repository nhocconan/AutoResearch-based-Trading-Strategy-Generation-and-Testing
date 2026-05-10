#!/usr/bin/env python3
# 4h_TRIX_ZeroCross_VolumeFilter_12hTrend
# Hypothesis: TRIX (Triple Exponential Average) zero-cross signals capture momentum shifts with low lag.
# Combined with 12h EMA trend filter and volume confirmation, this reduces false signals.
# TRIX > 0 indicates bullish momentum, < 0 bearish. We enter on zero-cross in trend direction.
# Volume confirmation ensures breakouts have participation. Designed for fewer trades (~25-40/year)
# to avoid fee drag, working in both bull (follow uptrend) and bear (follow downtrend) markets.

name = "4h_TRIX_ZeroCross_VolumeFilter_12hTrend"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate TRIX (15-period standard)
    # TRIX = EMA(EMA(EMA(close, 15), 15), 15) - then percent change
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean()
    ema2 = ema1.ewm(span=15, adjust=False, min_periods=15).mean()
    ema3 = ema2.ewm(span=15, adjust=False, min_periods=15).mean()
    trix = ema3.pct_change() * 100  # percentage
    trix_values = trix.values
    
    # Calculate 12h EMA50 for trend filter
    ema_50_12h = pd.Series(df_12h['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    # Volume confirmation (20-period MA on 4h = ~3.3 days)
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need TRIX (45 for three EMAs), 12h EMA50 (50), volume MA (20)
    start_idx = max(45, 50, 20)
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(trix_values[i]) or 
            np.isnan(ema_50_12h_aligned[i]) or 
            np.isnan(volume_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # 12h trend filter
        uptrend = close[i] > ema_50_12h_aligned[i]
        downtrend = close[i] < ema_50_12h_aligned[i]
        
        # Volume confirmation
        volume_confirm = volume[i] > volume_ma[i] * 1.5
        
        # TRIX zero-cross signals
        trix_now = trix_values[i]
        trix_prev = trix_values[i-1]
        
        bullish_cross = trix_prev <= 0 and trix_now > 0
        bearish_cross = trix_prev >= 0 and trix_now < 0
        
        if position == 0:
            # Long entry: bullish TRIX cross + uptrend + volume
            if bullish_cross and uptrend and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short entry: bearish TRIX cross + downtrend + volume
            elif bearish_cross and downtrend and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: bearish TRIX cross or trend breaks
            if bearish_cross or not uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: bullish TRIX cross or trend breaks
            if bullish_cross or not downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals