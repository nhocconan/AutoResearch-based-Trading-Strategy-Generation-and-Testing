#!/usr/bin/env python3
# 4h_1d_kama_rsi_volume_v2
# Strategy: 4h KAMA direction + RSI + volume confirmation + 1d volatility filter
# Timeframe: 4h
# Leverage: 1.0
# Hypothesis: KAMA adapts to market noise, reducing false signals in choppy markets.
# Long when KAMA rising, RSI < 60, volume above average, and 1d volatility low.
# Short when KAMA falling, RSI > 40, volume above average, and 1d volatility low.
# Uses 1d ATR percentile to filter high-volatility regimes where whipsaws occur.
# Designed for low trade frequency (~20-40/year) to minimize fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_kama_rsi_volume_v2"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price arrays
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1d ATR for volatility regime filter
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    # 1d ATR percentile (20-day lookback) to identify low-volatility regime
    atr_series = pd.Series(atr_1d)
    atr_percentile = atr_series.rolling(window=20, min_periods=20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    ).values
    atr_percentile_aligned = align_htf_to_ltf(prices, df_1d, atr_percentile)
    
    # KAMA (Kaufman Adaptive Moving Average) calculation
    def kama(close, er_length=10, fast_sc=2, slow_sc=30):
        change = np.abs(np.diff(close, prepend=close[0]))
        volatility = np.abs(np.diff(close)).rolling(window=er_length, min_periods=1).sum()
        er = np.where(volatility != 0, change / volatility, 0)
        sc = (er * (2/(fast_sc+1) - 2/(slow_sc+1)) + 2/(slow_sc+1)) ** 2
        kama = np.zeros_like(close)
        kama[0] = close[0]
        for i in range(1, len(close)):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        return kama
    
    kama_vals = kama(close)
    
    # RSI calculation
    def rsi(close, length=14):
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        avg_loss = pd.Series(loss).ewm(alpha=1/length, adjust=False, min_periods=length).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50).values
    
    rsi_vals = rsi(close)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_series = pd.Series(volume)
    vol_avg_20 = vol_series.rolling(window=20, min_periods=20).mean().values
    vol_confirm = volume > (1.5 * vol_avg_20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if any required data is invalid
        if (np.isnan(kama_vals[i]) or np.isnan(rsi_vals[i]) or 
            np.isnan(atr_percentile_aligned[i]) or np.isnan(vol_avg_20[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # KAMA direction
        kama_rising = kama_vals[i] > kama_vals[i-1]
        kama_falling = kama_vals[i] < kama_vals[i-1]
        
        # Entry conditions: KAMA direction + RSI + volume + low volatility
        if (kama_rising and rsi_vals[i] < 60 and vol_confirm[i] and 
            atr_percentile_aligned[i] < 40 and position != 1):
            position = 1
            signals[i] = 0.25
        elif (kama_falling and rsi_vals[i] > 40 and vol_confirm[i] and 
              atr_percentile_aligned[i] < 40 and position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: KAMA direction change or volatility spike
        elif position == 1 and (not kama_rising or atr_percentile_aligned[i] > 60):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (not kama_falling or atr_percentile_aligned[i] > 60):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            signals[i] = 0.25 if position == 1 else (-0.25 if position == -1 else 0.0)
    
    return signals