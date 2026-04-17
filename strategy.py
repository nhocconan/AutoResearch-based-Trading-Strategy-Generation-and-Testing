#!/usr/bin/env python3
"""
Hypothesis: 4h KAMA trend with 1d RSI filter and volume confirmation.
Uses Kaufman Adaptive Moving Average (KAMA) for trend direction, filtered by 1d RSI < 30 for long entries and > 70 for short entries.
Requires volume > 1.5x 20-day average for confirmation. Exits when KAMA reverses or volume drops.
Designed to capture trend reversals in both bull and bear markets with low trade frequency.
Target: 20-40 trades/year by requiring confluence of KAMA direction, RSI extreme, and volume.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # === KAMA on 4h (trend) ===
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # Calculate KAMA (ER=10, fast=2, slow=30)
    change = np.abs(np.diff(close_4h, prepend=close_4h[0]))
    volatility = np.sum(np.abs(np.diff(close_4h)), axis=0)  # placeholder, will compute properly below
    # Recalculate volatility properly
    volatility = np.zeros_like(close_4h)
    for i in range(1, len(close_4h)):
        volatility[i] = volatility[i-1] + np.abs(close_4h[i] - close_4h[i-1])
    
    er = np.zeros_like(close_4h)
    for i in range(len(close_4h)):
        if volatility[i] > 0:
            er[i] = change[i] / volatility[i]
        else:
            er[i] = 0
    
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    # Alternative simpler KAMA approximation using EMA of ER
    # Using pandas for simplicity and correctness
    close_4h_series = pd.Series(close_4h)
    change = close_4h_series.diff().abs()
    volatility = change.rolling(window=10, min_periods=1).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)
    sc = (er * (2/2 - 2/30) + 2/30) ** 2
    kama = [close_4h[0]]
    for i in range(1, len(close_4h)):
        kama.append(kama[-1] + sc.iloc[i] * (close_4h[i] - kama[-1]))
    kama = np.array(kama)
    
    kama_4h = kama
    kama_4h_series = pd.Series(kama_4h)
    kama_slope = kama_4h_series.diff(periods=3)  # 3-period slope for trend
    
    # Align KAMA slope to 4h
    kama_slope_aligned = align_htf_to_ltf(prices, df_4h, kama_slope.values)
    
    # === 1d RSI Filter ===
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50).values
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # === 1d Volume Confirmation ===
    volume_1d = df_1d['volume'].values
    vol_ma_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_ma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20_1d)
    vol_today_aligned = align_htf_to_ltf(prices, df_1d, volume_1d)
    
    signals = np.zeros(n)
    
    # Warmup
    warmup = 60
    
    # Track position
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any data is NaN
        if (np.isnan(kama_slope_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or
            np.isnan(vol_ma_20_1d_aligned[i]) or np.isnan(vol_today_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-day average
        vol_confirm = vol_today_aligned[i] > vol_ma_20_1d_aligned[i] * 1.5
        
        # KAMA trend: positive slope = uptrend, negative = downtrend
        kama_up = kama_slope_aligned[i] > 0
        kama_down = kama_slope_aligned[i] < 0
        
        # RSI extremes: <30 oversold, >70 overbought
        rsi_oversold = rsi_1d_aligned[i] < 30
        rsi_overbought = rsi_1d_aligned[i] > 70
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: KAMA up + RSI oversold + volume confirmation
            if kama_up and rsi_oversold and vol_confirm:
                signals[i] = 0.25
                position = 1
                continue
            # Short: KAMA down + RSI overbought + volume confirmation
            elif kama_down and rsi_overbought and vol_confirm:
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic
        elif position == 1:
            # Exit long: KAMA turns down or volume fails
            if kama_down or not vol_confirm:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: KAMA turns up or volume fails
            if kama_up or not vol_confirm:
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_KAMA_1dRSI_VolumeConfirm"
timeframe = "4h"
leverage = 1.0