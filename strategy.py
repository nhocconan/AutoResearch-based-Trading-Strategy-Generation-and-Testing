#!/usr/bin/env python3
"""
Hypothesis: 12h KAMA trend + 1d RSI mean reversion + volume spike confirmation.
- Uses 12h timeframe (primary) and 1d HTF for RSI(14) extreme levels (proven BTC/ETH edge in ranging markets)
- KAMA(ER=10, SC=2) on 12h closes determines trend direction (adaptive, low-lag)
- Long when price > 12h KAMA AND 1d RSI < 30 (oversold) AND volume > 1.5 * volume MA(20)
- Short when price < 12h KAMA AND 1d RSI > 70 (overbought) AND volume > 1.5 * volume MA(20)
- Exit when price crosses back below/above 12h KAMA (trend change)
- Discrete signal size: 0.25 to minimize fee churn
- Target: 50-150 total trades over 4 years (12-37/year) as per 12h timeframe recommendation
- Works in both bull/bear: RSI mean reversion captures corrections in trends, KAMA avoids whipsaws
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h KAMA for trend (ER=10, SC=2 - standard settings)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 2:  # Need at least 2 points for KAMA
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_12h, n=10))  # 10-period net change
    volatility = np.sum(np.abs(np.diff(close_12h, n=1)), axis=1)  # 10-period sum of abs changes
    # Pad the beginning with NaN for alignment
    change_padded = np.concatenate([np.full(9, np.nan), change])
    volatility_padded = np.concatenate([np.full(9, np.nan), volatility])
    er = np.where(volatility_padded > 0, change_padded / volatility_padded, 0)
    # Smoothing Constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # ER-based smoothing constant
    # Calculate KAMA
    kama = np.full_like(close_12h, np.nan)
    kama[0] = close_12h[0]  # Initialize with first price
    for i in range(1, len(close_12h)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_12h[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_aligned = align_htf_to_ltf(prices, df_12h, kama)
    
    # Calculate 1d RSI(14) for mean reversion
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 15:  # Need enough data for RSI(14)
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    # First average gain/loss
    avg_gain = np.mean(gain[:14]) if len(gain) >= 14 else np.nan
    avg_loss = np.mean(loss[:14]) if len(loss) >= 14 else np.nan
    rs = np.full_like(close_1d, np.nan)
    rsi = np.full_like(close_1d, np.nan)
    if not np.isnan(avg_loss) and avg_loss != 0:
        rs[13] = avg_gain / avg_loss
        rsi[13] = 100 - (100 / (1 + rs[13]))
    elif not np.isnan(avg_loss) and avg_loss == 0:
        rs[13] = np.inf
        rsi[13] = 100
    # Subsequent values using Wilder's smoothing
    for i in range(14, len(close_1d)):
        avg_gain = (avg_gain * 13 + gain[i-1]) / 14
        avg_loss = (avg_loss * 13 + loss[i-1]) / 14
        if avg_loss != 0:
            rs[i] = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rs[i] = np.inf
            rsi[i] = 100
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: current volume > 1.5 * 20-period volume MA
    volume_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * volume_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(14, 20)  # Need RSI(14), volume MA(20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(volume_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price > KAMA AND RSI < 30 (oversold) AND volume confirmation
            if close[i] > kama_aligned[i] and rsi_aligned[i] < 30 and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < KAMA AND RSI > 70 (overbought) AND volume confirmation
            elif close[i] < kama_aligned[i] and rsi_aligned[i] > 70 and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: price crosses below KAMA (trend change)
            if close[i] < kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: price crosses above KAMA (trend change)
            if close[i] > kama_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_1dRSI_VolumeConfirm_v1"
timeframe = "12h"
leverage = 1.0