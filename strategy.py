#!/usr/bin/env python3
name = "4h_KAMA_RSI_Trend_Volume_Filter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    close_1d = df_1d['close'].values
    er = np.zeros_like(close_1d)
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.sum(np.abs(np.diff(close_1d, prepend=close_1d[0])), axis=0)  # This needs fixing
    # Let's recalculate properly
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        volatility[i] = volatility[i-1] + np.abs(close_1d[i] - close_1d[i-1])
    volatility = np.maximum(volatility, 1e-10)  # avoid division by zero
    er = change / volatility
    er[0] = 0  # first value
    
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2  # fast=2, slow=30
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate RSI on daily close
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Align KAMA and RSI to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Load 1h data for volume filter (more responsive than 4h volume)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 20:
        return np.zeros(n)
    
    # Volume MA on 1h data
    vol_1h = df_1h['volume'].values
    vol_ma_1h = pd.Series(vol_1h).rolling(window=20, min_periods=20).mean().values
    vol_ma_1h_aligned = align_htf_to_ltf(prices, df_1h, vol_ma_1h)
    
    # Current 4h volume
    vol_4h = volume
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Warmup period
    
    for i in range(start_idx, n):
        if np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_1h_aligned[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: current 4h volume > 1.5x 1h volume MA
        vol_filter = vol_4h[i] > 1.5 * vol_ma_1h_aligned[i]
        
        if position == 0:
            # Long: Price > KAMA, RSI > 50, volume filter
            if close[i] > kama_aligned[i] and rsi_aligned[i] > 50 and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: Price < KAMA, RSI < 50, volume filter
            elif close[i] < kama_aligned[i] and rsi_aligned[i] < 50 and vol_filter:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: Price < KAMA or RSI < 40
            if close[i] < kama_aligned[i] or rsi_aligned[i] < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: Price > KAMA or RSI > 60
            if close[i] > kama_aligned[i] or rsi_aligned[i] > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h strategy using daily KAMA for trend direction and daily RSI for momentum,
# filtered by 1h volume spikes to avoid low-activity periods. KAMA adapts to market noise,
# providing better trend detection than traditional MA in volatile crypto markets.
# RSI filters avoid overextended moves. Volume confirmation ensures trades occur
# during active participation. Position size 0.25 limits drawdown. Target: 20-40 trades/year.