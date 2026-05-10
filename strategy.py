#!/usr/bin/env python3
# 6h_KAMA_Trend_Reversal
# Hypothesis: KAMA adapts to market noise, reducing whipsaw in sideways markets.
# Long when price crosses above KAMA(10) with volume > 1.5x average and price > 1w EMA50 (bullish regime).
# Short when price crosses below KAMA(10) with volume > 1.5x average and price < 1w EMA50 (bearish regime).
# Exit when price crosses KAMA in opposite direction.
# Designed for 20-50 trades/year to avoid fee drag.

name = "6h_KAMA_Trend_Reversal"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Efficiency Ratio (ER) for KAMA
    change = np.abs(np.diff(close, n=10))  # 10-period change
    vol = np.sum(np.abs(np.diff(close)), axis=1)  # 10-period volatility
    # Pad arrays to match length
    change = np.concatenate([[np.nan]*10, change])
    vol = np.concatenate([[np.nan]*10, vol])
    er = np.where(vol != 0, change / vol, 0)
    
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    
    # KAMA calculation
    kama = np.full(n, np.nan)
    kama[0] = close[0]
    for i in range(1, n):
        if np.isnan(sc[i]):
            kama[i] = kama[i-1]
        else:
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # Get 1w EMA50 for regime filter
    df_1w = get_htf_data(prices, '1w')
    ema_50_1w = pd.Series(df_1w['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume average (20 periods)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.nanmean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 30  # Ensure sufficient warmup
    
    for i in range(start_idx, n):
        if np.isnan(kama[i]) or np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Determine regime based on 1w EMA50
            if close[i] > ema_50_1w_aligned[i]:  # Bullish regime
                # Long: price crosses above KAMA with volume confirmation
                if close[i] > kama[i] and close[i-1] <= kama[i-1] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = 0.25
                    position = 1
            else:  # Bearish regime
                # Short: price crosses below KAMA with volume confirmation
                if close[i] < kama[i] and close[i-1] >= kama[i-1] and volume[i] > 1.5 * vol_ma[i]:
                    signals[i] = -0.25
                    position = -1
        
        elif position == 1:
            # Exit: price crosses below KAMA
            if close[i] < kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit: price crosses above KAMA
            if close[i] > kama[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals