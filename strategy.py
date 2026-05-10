#!/usr/bin/env python3
"""
6h_Stochastic_Trend_1w_1d
Hypothesis: Combines stochastic oscillator with weekly trend and daily volume confirmation.
Stochastic identifies overbought/oversold conditions in ranging markets, while the 1-week EMA200
defines the long-term trend direction. In strong trends, we trade pullbacks; in ranging markets,
we mean-revert at extremes. Volume confirmation filters weak signals. This dual approach works
in both bull (buy dips in uptrend) and bear (sell rallies in downtrend) markets.
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "6h_Stochastic_Trend_1w_1d"
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
    
    # 1w EMA200 for long-term trend
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema200_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 200:
        ema200_1w[199] = np.mean(close_1w[:200])
        alpha = 2 / (200 + 1)
        for i in range(200, len(close_1w)):
            ema200_1w[i] = alpha * close_1w[i] + (1 - alpha) * ema200_1w[i-1]
    ema200_1w_aligned = align_htf_to_ltf(prices, df_1w, ema200_1w)
    
    # 1d volume SMA20 for confirmation
    df_1d = get_htf_data(prices, '1d')
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # Stochastic oscillator (14,3,3) on 6h data
    k_period = 14
    d_period = 3
    lowest_low = np.full(n, np.nan)
    highest_high = np.full(n, np.nan)
    
    for i in range(n):
        if i >= k_period - 1:
            lowest_low[i] = np.min(low[i - k_period + 1:i + 1])
            highest_high[i] = np.max(high[i - k_period + 1:i + 1])
    
    stoch_k = np.full(n, np.nan)
    stoch_d = np.full(n, np.nan)
    
    for i in range(n):
        if highest_high[i] - lowest_low[i] != 0:
            stoch_k[i] = 100 * (close[i] - lowest_low[i]) / (highest_high[i] - lowest_low[i])
    
    for i in range(n):
        if i >= d_period - 1 and not np.isnan(stoch_k[i - d_period + 1:i + 1]).any():
            stoch_d[i] = np.mean(stoch_k[i - d_period + 1:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(k_period + d_period, 200, 20)
    
    for i in range(start_idx, n):
        if np.isnan(ema200_1w_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or \
           np.isnan(stoch_k[i]) or np.isnan(stoch_d[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 6h volume > 1.2x average 1d volume (scaled to 6h)
        vol_6h_approx = vol_sma20_1d_aligned[i] / 4.0
        volume_confirm = volume[i] > 1.2 * vol_6h_approx
        
        if position == 0:
            # Long: Oversold in uptrend or overbought in downtrend (mean reversion)
            if stoch_k[i] < 20 and stoch_d[i] < 20 and close[i] > ema200_1w_aligned[i] and volume_confirm:
                signals[i] = 0.25
                position = 1
            elif stoch_k[i] > 80 and stoch_d[i] > 80 and close[i] < ema200_1w_aligned[i] and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Stochastic becomes overbought or trend reversal
            if stoch_k[i] > 80 or close[i] < ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Stochastic becomes oversold or trend reversal
            if stoch_k[i] < 20 or close[i] > ema200_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals