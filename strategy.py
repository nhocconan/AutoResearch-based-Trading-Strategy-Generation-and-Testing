#!/usr/bin/env python3
"""
1d_1w_KAMA_Direction_WeeklyTrend
Hypothesis: Uses daily KAMA direction filtered by weekly trend (EMA34) and volume confirmation.
Trades only in direction of weekly trend to avoid counter-trend losses. Designed for low turnover
(10-30 trades/year) with high conviction signals to minimize fee drag and perform in both bull and bear.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # Calculate weekly EMA34 for trend filter
    close_1w = df_1w['close'].values
    ema_34_1w = np.zeros_like(close_1w)
    ema_34_1w[:] = np.nan
    if len(close_1w) >= 34:
        k = 2 / (34 + 1)
        ema_34_1w[33] = np.mean(close_1w[:34])
        for i in range(34, len(close_1w)):
            ema_34_1w[i] = close_1w[i] * k + ema_34_1w[i-1] * (1 - k)
    
    # Align weekly EMA34 to daily timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate daily KAMA (ER=10)
    change = np.abs(np.diff(close, prepend=close[0]))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Use expanding window for ER calculation to avoid look-ahead
    er = np.zeros_like(change)
    for i in range(len(change)):
        if i < 10:
            er[i] = 0.0
        else:
            price_change = np.abs(close[i] - close[i-10])
            price_volatility = np.sum(np.abs(np.diff(close[i-10:i+1])))
            er[i] = price_change / price_volatility if price_volatility > 0 else 0
    
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Volume confirmation: current volume > 1.5x 20-day average
    vol_ma = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma[i] = np.mean(volume[0:i+1]) if i >= 0 else volume[i]
        else:
            vol_ma[i] = np.mean(volume[i-20+1:i+1])
    vol_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Warmup for KAMA and EMA
    
    for i in range(start_idx, n):
        if np.isnan(ema_34_1w_aligned[i]) or np.isnan(kama[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA trending up, above weekly EMA34, with volume spike
            if kama[i] > kama[i-1] and close[i] > ema_34_1w_aligned[i] and vol_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: KAMA trending down, below weekly EMA34, with volume spike
            elif kama[i] < kama[i-1] and close[i] < ema_34_1w_aligned[i] and vol_spike[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit: KAMA turns down or breaks below weekly EMA34
            if kama[i] < kama[i-1] or close[i] < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: KAMA turns up or breaks above weekly EMA34
            if kama[i] > kama[i-1] or close[i] > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_1w_KAMA_Direction_WeeklyTrend"
timeframe = "1d"
leverage = 1.0