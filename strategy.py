#!/usr/bin/env python3
"""
6h_RelativeVigorIndex_1wTrend_Filter
Hypothesis: RVI (Relative Vigor Index) captures trend strength via close-open relative to high-low range.
Long when RVI > 0 and rising, short when RVI < 0 and falling, filtered by weekly trend (price > weekly EMA50).
Exit on RVI mean reversion (crossing zero) or trend change. Uses volume confirmation to avoid false signals.
Designed for low turnover (~20-40 trades/year) to minimize fee drag on 6B timeframe.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    ema50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # RVI calculation: (Close - Open) / (High - Low) smoothed
    # Numerator: close - open
    num = close - open_
    # Denominator: high - low (avoid division by zero)
    den = high - low
    den = np.where(den == 0, 1e-10, den)  # small epsilon to prevent div/0
    raw_rvi = num / den
    
    # Smooth with 4-period SMA (standard RVI uses 4)
    rvi_raw = pd.Series(raw_rvi).rolling(window=4, min_periods=4).mean().values
    # Signal line: 4-period SMA of RVI
    rvi_signal = pd.Series(rvi_raw).rolling(window=4, min_periods=4).mean().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for RVI smoothing and weekly EMA
    start_idx = max(20, 10)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if np.isnan(rvi_signal[i]) or np.isnan(rvi_raw[i]) or np.isnan(ema50_1w_aligned[i]):
            signals[i] = 0.0
            continue
        
        rvi_val = rvi_signal[i]
        rvi_raw_val = rvi_raw[i]
        ema_trend = ema50_1w_aligned[i]
        vol_conf = vol_confirm[i]
        
        if position == 0:
            # Long: RVI rising above zero + volume confirmation + uptrend (price > weekly EMA50)
            if rvi_raw_val > 0 and rvi_val > rvi_raw_val and vol_conf and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: RVI falling below zero + volume confirmation + downtrend (price < weekly EMA50)
            elif rvi_raw_val < 0 and rvi_val < rvi_raw_val and vol_conf and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: RVI crosses below zero or trend turns down
            if rvi_raw_val <= 0 or close[i] < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: RVI crosses above zero or trend turns up
            if rvi_raw_val >= 0 or close[i] > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_RelativeVigorIndex_1wTrend_Filter"
timeframe = "6h"
leverage = 1.0