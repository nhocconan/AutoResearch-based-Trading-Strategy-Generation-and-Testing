#!/usr/bin/env python3
# 4h_1d_atr_breakout_v2
# Hypothesis: 4-hour ATR-based breakout with 1-day trend filter and volume confirmation.
# Long when price breaks above ATR(14) upper band, volume > 1.2x average, and price above EMA50(1d).
# Short when price breaks below ATR(14) lower band, volume > 1.2x average, and price below EMA50(1d).
# Exit when price returns to ATR(14) midline.
# Designed to generate ~25-35 trades/year with strong risk-reward to avoid fee decay.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_atr_breakout_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # ATR(14) calculation
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr_sum = 0.0
    for i in range(n):
        if i < 14:
            atr_sum += tr[i]
            if i == 13:
                atr[i] = atr_sum / 14
        else:
            atr_sum = atr_sum - atr_sum/14 + tr[i]
            atr[i] = atr_sum / 14
    
    # ATR bands
    atr_upper = np.full(n, np.nan)
    atr_lower = np.full(n, np.nan)
    atr_mid = np.full(n, np.nan)
    for i in range(14, n):
        atr_upper[i] = close[i-1] + 1.5 * atr[i]
        atr_lower[i] = close[i-1] - 1.5 * atr[i]
        atr_mid[i] = close[i-1]
    
    # Volume MA(20)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    # Get 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        alpha = 2.0 / (50 + 1)
        for i in range(50, len(close_1d)):
            ema50_1d[i] = alpha * close_1d[i] + (1 - alpha) * ema50_1d[i-1]
    
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        if (np.isnan(atr_upper[i]) or np.isnan(atr_lower[i]) or np.isnan(atr_mid[i]) or
            np.isnan(vol_ma[i]) or np.isnan(ema50_1d_aligned[i])):
            if position != 0:
                pass
            else:
                signals[i] = 0.0
            continue
        
        vol_ratio = volume[i] / vol_ma[i] if vol_ma[i] > 0 else 0
        price = close[i]
        
        if position == 1:
            if price <= atr_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:
            if price >= atr_mid[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:
            if price > atr_upper[i] and vol_ratio > 1.2 and price > ema50_1d_aligned[i]:
                position = 1
                signals[i] = 0.25
            elif price < atr_lower[i] and vol_ratio > 1.2 and price < ema50_1d_aligned[i]:
                position = -1
                signals[i] = -0.25
    
    return signals