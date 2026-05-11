#!/usr/bin/env python3
"""
12h_1d_KAMA_Trend_With_RSI_Filter
Hypothesis: Use KAMA on 1d for trend direction, RSI(14) for momentum filter, and volume spike for entry confirmation on 12h. Enter long when KAMA up, RSI < 40, and volume > 1.5x average; short when KAMA down, RSI > 60, and volume > 1.5x average. Exit on opposite RSI extreme or ATR stop. Designed for low trade frequency (<30/year) and robustness in both bull and bear markets via trend + momentum filters.
"""

name = "12h_1d_KAMA_Trend_With_RSI_Filter"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for trend and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 12h OHLCV
    close_12h = prices['close'].values
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    volume_12h = prices['volume'].values
    
    # --- 1d KAMA Trend ---
    close_1d = df_1d['close'].values
    # Efficiency Ratio
    change = np.abs(np.diff(close_1d, k=10))  # 10-period change
    abs_change = np.abs(np.diff(close_1d, k=1))
    er_num = np.concatenate([[np.nan]*10, change])
    er_den = np.concatenate([[np.nan]*10, np.cumsum(abs_change)[9:]])
    er = np.where(er_den != 0, er_num / er_den, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.full_like(close_1d, np.nan)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    kama_1d = kama
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # --- 1d RSI(14) ---
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi = np.concatenate([[np.nan]*14, rsi[14:]])  # align with original
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # --- Volume Spike: > 1.5x 20-period average ---
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=10).mean().values
    vol_spike = volume_12h > (1.5 * vol_ma)
    
    # --- ATR for stoploss (14-period) ---
    tr1 = np.abs(high_12h - low_12h)
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    start_idx = 30  # for RSI and KAMA
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                if position == 1 and close_12h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif position == -1 and close_12h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25 if position == 1 else -0.25
            continue
        
        kama_up = close_12h[i] > kama_1d_aligned[i]
        kama_down = close_12h[i] < kama_1d_aligned[i]
        rsi_oversold = rsi_1d_aligned[i] < 40
        rsi_overbought = rsi_1d_aligned[i] > 60
        vol_ok = vol_spike[i]
        
        if position == 0:
            if kama_up and rsi_oversold and vol_ok:
                signals[i] = 0.25
                position = 1
                entry_price = close_12h[i]
            elif kama_down and rsi_overbought and vol_ok:
                signals[i] = -0.25
                position = -1
                entry_price = close_12h[i]
        else:
            if position == 1:
                if close_12h[i] <= entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif rsi_1d_aligned[i] > 60:  # exit on RSI overbought
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                if close_12h[i] >= entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                elif rsi_1d_aligned[i] < 40:  # exit on RSI oversold
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals