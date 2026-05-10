#!/usr/bin/env python3
# 6h_MarketFacilitationIndex_1dTrend_Filter
# Hypothesis: MFI (Market Facilitation Index) = (High - Low) / Volume measures price movement efficiency.
# Rising MFI with increasing volume indicates strong trend continuation.
# Falling MFI with increasing volume indicates weakening trend (fade signal).
# Combined with 1-day trend filter (EMA50) to align with higher timeframe direction.
# Works in both bull/bear markets by following 1d trend while using MMI for entry/exit timing.
# Targets 15-35 trades per year on 6h timeframe with position size 0.25.

name = "6h_MarketFacilitationIndex_1dTrend_Filter"
timeframe = "6h"
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
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend direction
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate MFI components
    price_range = high - low  # High - Low
    # Avoid division by zero - add small epsilon where volume is zero
    volume_safe = np.where(volume == 0, 1e-10, volume)
    mfi = price_range / volume_safe
    
    # Calculate 3-period EMA of MFI for smoothing (to reduce noise)
    mfi_series = pd.Series(mfi)
    mfi_ema = mfi_series.ewm(span=3, adjust=False).fillna(0).values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup for 1d EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1d
        uptrend = close[i] > ema_50_1d_aligned[i]
        downtrend = close[i] < ema_50_1d_aligned[i]
        
        if position == 0:
            # Long entry: MFI rising AND 1-day uptrend
            # MFI rising: current MFI > previous MFI
            if i > 0 and mfi_ema[i] > mfi_ema[i-1] and uptrend:
                signals[i] = 0.25
                position = 1
            # Short entry: MFI falling AND 1-day downtrend
            # MFI falling: current MFI < previous MFI
            elif i > 0 and mfi_ema[i] < mfi_ema[i-1] and downtrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: MFI falling OR 1-day trend turns down
            if i > 0 and mfi_ema[i] < mfi_ema[i-1] or downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: MFI rising OR 1-day trend turns up
            if i > 0 and mfi_ema[i] > mfi_ema[i-1] or uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals