#!/usr/bin/env python3
"""
12h_KAMA_Trend_1dRSI_Volume
Hypothesis: KAMA trend direction on 12h with 1d RSI filter and volume confirmation.
KAMA adapts to market noise, providing reliable trend signals.
In trending markets, KAMA follows price closely; in ranging markets, it stays flat.
RSI filter avoids overbought/oversold extremes. Volume confirmation ensures strong breakouts.
Works in both bull (KAMA up + RSI<70) and bear (KAMA down + RSI>30).
Target: 50-150 total trades over 4 years (12-37/year).
"""

name = "12h_KAMA_Trend_1dRSI_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 1d RSI(14) for overbought/oversold filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    rsi_14_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 14:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.full(len(close_1d), np.nan)
        avg_loss = np.full(len(close_1d), np.nan)
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, np.inf)
        rsi_14_1d = 100 - (100 / (1 + rs))
        rsi_14_1d[np.isinf(rs)] = 100
    rsi_14_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_1d)
    
    # 1d volume SMA20 for volume confirmation
    volume_1d = df_1d['volume'].values
    vol_sma20_1d = np.full(len(volume_1d), np.nan)
    if len(volume_1d) >= 20:
        vol_sma20_1d[19] = np.mean(volume_1d[:20])
        for i in range(20, len(volume_1d)):
            vol_sma20_1d[i] = (vol_sma20_1d[i-1] * 19 + volume_1d[i]) / 20
    vol_sma20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma20_1d)
    
    # KAMA(10) on 12h for trend direction
    er_period = 10
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    kama = np.full(n, np.nan)
    if n >= er_period:
        change = np.abs(close[er_period:] - close[:-er_period])
        volatility = np.sum(np.abs(np.diff(close)), axis=0)  # placeholder, will compute properly
        # Correct volatility calculation
        volatility = np.array([np.sum(np.abs(np.diff(close[i:i+er_period]))) for i in range(n-er_period+1)])
        volatility = np.concatenate([np.full(er_period-1, np.nan), volatility])
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
        kama[er_period-1] = np.mean(close[:er_period])
        for i in range(er_period, n):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(14, er_period)  # warmup for RSI and KAMA
    
    for i in range(start_idx, n):
        if np.isnan(rsi_14_1d_aligned[i]) or np.isnan(vol_sma20_1d_aligned[i]) or np.isnan(kama[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 12h volume > 1.5x average 1d volume (scaled to 12h)
        # Approximate 12h volume from 1d: 1d volume / 2 (since 24h/12h = 2)
        vol_12h_approx = vol_sma20_1d_aligned[i] / 2.0
        volume_confirm = volume[i] > 1.5 * vol_12h_approx
        
        if position == 0:
            # Long: KAMA rising (bullish trend) with RSI not overbought and volume confirmation
            if kama[i] > kama[i-1] and rsi_14_1d_aligned[i] < 70 and volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling (bearish trend) with RSI not oversold and volume confirmation
            elif kama[i] < kama[i-1] and rsi_14_1d_aligned[i] > 30 and volume_confirm:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: KAMA reverses or RSI overbought
            if kama[i] < kama[i-1] or rsi_14_1d_aligned[i] >= 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: KAMA reverses or RSI oversold
            if kama[i] > kama[i-1] or rsi_14_1d_aligned[i] <= 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals