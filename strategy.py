#!/usr/bin/env python3
"""
6h_RankedMomentum_1dTrend_Volume
Hypothesis: Rank momentum (percentile of return over 6 periods) identifies strong trending moves.
Enter long when rank momentum > 80 and price above 1d EMA50 with volume confirmation.
Enter short when rank momentum < 20 and price below 1d EMA50 with volume confirmation.
Exit when rank momentum crosses back to neutral (40-60 range).
Uses 1d trend filter and volume spike to avoid false signals. Designed for 6h timeframe
to capture medium-term momentum in both bull and bear markets. Target: 20-40 trades/year.
"""

name = "6h_RankedMomentum_1dTrend_Volume"
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
    
    # 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50_1d[49] = np.mean(close_1d[:50])
        alpha = 2 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema_50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema_50_1d[i-1]
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma_20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma_20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma_20_1d[i] = (vol_sma_20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # Rank momentum: percentile rank of returns over 6 periods (36 hours)
    lookback = 6
    returns = np.diff(np.log(close), prepend=np.log(close[0]))
    rank_momentum = np.full(n, np.nan)
    
    for i in range(lookback, n):
        window_returns = returns[i-lookback:i+1]
        # Rank current return within the window (0 to 100)
        current_return = returns[i]
        rank = np.sum(window_returns <= current_return) / len(window_returns) * 100
        rank_momentum[i] = rank
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, lookback)  # warmup for EMA50 and lookback
    
    for i in range(start_idx, n):
        if np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20_1d_aligned[i]) or \
           np.isnan(rank_momentum[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.5x average 6h volume (scaled from 1d)
        # Approximate 6h volume from 1d: 1d volume / 4 (since 24h/6h = 4)
        vol_6h_approx = vol_sma_20_1d_aligned[i] / 4.0
        volume_confirm = volume[i] > 1.5 * vol_6h_approx
        
        if position == 0:
            # Long: Strong upward momentum + uptrend + volume
            if rank_momentum[i] > 80 and close[i] > ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: Strong downward momentum + downtrend + volume
            elif rank_momentum[i] < 20 and close[i] < ema_50_1d_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Momentum weakens or trend turns
            if rank_momentum[i] < 40 or close[i] < ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Momentum weakens or trend turns
            if rank_momentum[i] > 60 or close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals