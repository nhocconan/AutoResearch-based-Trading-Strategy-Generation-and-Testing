#!/usr/bin/env python3
"""
4H_DonchianBreakout_TrendVolatilityFilter
Hypothesis: Uses 4h Donchian channel breakouts with 1d trend filter (EMA34) and volatility filter (ATR ratio).
Enters long when price breaks above upper Donchian(20) in a 1d uptrend with expanding volatility,
and short when price breaks below lower Donchian(20) in a 1d downtrend with expanding volatility.
Designed for low trade frequency (target: 25-40/year) to avoid fee drag while capturing strong trends.
"""

name = "4H_DonchianBreakout_TrendVolatilityFilter"
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
    
    # Donchian channels on 4h
    donchian_period = 20
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    for i in range(donchian_period - 1, n):
        upper[i] = np.max(high[i-donchian_period+1:i+1])
        lower[i] = np.min(low[i-donchian_period+1:i+1])
    
    # 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    ema34_1d = np.full_like(close_1d, np.nan)
    if len(close_1d) >= 34:
        ema34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema34_1d[i] = (close_1d[i] * 2/35) + (ema34_1d[i-1] * 33/35)
    
    # ATR for volatility filter
    atr_period = 14
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    if n >= atr_period:
        atr[atr_period-1] = np.mean(tr[:atr_period])
        for i in range(atr_period, n):
            atr[i] = (tr[i] + (atr_period-1) * atr[i-1]) / atr_period
    
    # ATR ratio (current ATR / 20-period ATR average) for volatility filter
    atr_ma_period = 20
    atr_ma = np.full(n, np.nan)
    for i in range(atr_ma_period-1, n):
        atr_ma[i] = np.mean(atr[i-atr_ma_period+1:i+1])
    
    atr_ratio = np.full(n, np.nan)
    for i in range(n):
        if not np.isnan(atr[i]) and not np.isnan(atr_ma[i]) and atr_ma[i] > 0:
            atr_ratio[i] = atr[i] / atr_ma[i]
    
    # Align 1d indicators to 4h
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(donchian_period, 34, atr_period, atr_ma_period)
    
    for i in range(start_idx, n):
        if np.isnan(upper[i]) or np.isnan(lower[i]) or np.isnan(ema34_1d_aligned[i]) or np.isnan(atr_ratio[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend condition: price vs 1d EMA34
        is_uptrend = close[i] > ema34_1d_aligned[i]
        is_downtrend = close[i] < ema34_1d_aligned[i]
        
        # Volatility filter: expanding volatility (ATR ratio > 1.1)
        vol_expanding = atr_ratio[i] > 1.1
        
        if position == 0:
            # Long: break above upper Donchian in uptrend with expanding volatility
            if is_uptrend and vol_expanding and close[i] > upper[i]:
                signals[i] = 0.25
                position = 1
            # Short: break below lower Donchian in downtrend with expanding volatility
            elif is_downtrend and vol_expanding and close[i] < lower[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: trend reversal or volatility contraction
            if not is_uptrend or not vol_expanding:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: trend reversal or volatility contraction
            if not is_downtrend or not vol_expanding:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals