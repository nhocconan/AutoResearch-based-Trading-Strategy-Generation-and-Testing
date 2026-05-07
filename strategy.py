#!/usr/bin/env python3
name = "4h_1d_KAMA_RSI_Chop_Filter"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily KAMA: Kaufman Adaptive Moving Average
    close_1d = df_1d['close'].values
    direction_1d = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility_1d = np.abs(np.diff(close_1d))
    er_1d = np.where(volatility_1d > 0, direction_1d / volatility_1d, 0)
    sc_1d = (er_1d * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    kama_1d = np.zeros_like(close_1d)
    kama_1d[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama_1d[i] = kama_1d[i-1] + sc_1d[i] * (close_1d[i] - kama_1d[i-1])
    kama_1d = np.where(np.arange(len(close_1d)) < 30, np.nan, kama_1d)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Daily RSI(14)
    delta_1d = np.diff(close_1d, prepend=close_1d[0])
    gain_1d = np.where(delta_1d > 0, delta_1d, 0)
    loss_1d = np.where(delta_1d < 0, -delta_1d, 0)
    avg_gain_1d = pd.Series(gain_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss_1d = pd.Series(loss_1d).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs_1d = np.where(avg_loss_1d != 0, avg_gain_1d / avg_loss_1d, 0)
    rsi_1d = 100 - (100 / (1 + rs_1d))
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Weekly Chop Index (14) for regime filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 14:
        return np.zeros(n)
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    atr_1w = np.zeros(len(high_1w))
    for i in range(1, len(high_1w)):
        tr = max(high_1w[i] - low_1w[i], 
                 abs(high_1w[i] - high_1w[i-1]), 
                 abs(low_1w[i] - low_1w[i-1]))
        atr_1w[i] = tr if i == 1 else (atr_1w[i-1] * 13 + tr) / 14
    highest_1w = pd.Series(high_1w).rolling(window=14, min_periods=14).max().values
    lowest_1w = pd.Series(low_1w).rolling(window=14, min_periods=14).min().values
    chop_1w = 100 * np.log10((highest_1w - lowest_1w) / np.sum(np.abs(np.diff(high_1w, prepend=high_1w[0])) + 1e-10, axis=0)) / np.log10(14)
    chop_1w = np.where(np.arange(len(high_1w)) < 14, np.nan, chop_1w)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w, additional_delay_bars=2)
    
    # Volume spike detection: 6-period average (1.5 days of 4h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 6)  # Wait for KAMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1w_aligned[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above KAMA, RSI > 50, chop > 61.8 (range) -> mean reversion long
            vol_condition = volume[i] > vol_ma_6[i] * 1.8
            if close[i] > kama_1d_aligned[i] and rsi_1d_aligned[i] > 50 and chop_1w_aligned[i] > 61.8 and vol_condition:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA, RSI < 50, chop > 61.8 (range) -> mean reversion short
            elif close[i] < kama_1d_aligned[i] and rsi_1d_aligned[i] < 50 and chop_1w_aligned[i] > 61.8 and vol_condition:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below KAMA or RSI < 40 or chop < 38.2 (trend)
            if close[i] < kama_1d_aligned[i] or rsi_1d_aligned[i] < 40 or chop_1w_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above KAMA or RSI > 60 or chop < 38.2 (trend)
            if close[i] > kama_1d_aligned[i] or rsi_1d_aligned[i] > 60 or chop_1w_aligned[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h mean reversion using daily KAMA trend + RSI + weekly chop filter
# - Daily KAMA acts as adaptive trend: price > KAMA = bullish bias, < KAMA = bearish
# - Daily RSI > 50 for longs, < 50 for shorts ensures momentum alignment
# - Weekly Chop > 61.8 indicates ranging market ideal for mean reversion
# - Volume spike (1.8x average) confirms institutional participation at extremes
# - Works in both bull and bear markets by adapting to weekly regime
# - Exit when price returns to KAMA, RSI reverses, or market trends (chop < 38.2)
# - Position size 0.25 targets ~30-50 trades/year, avoiding fee drag
# - Uses actual daily/weekly data via mtf_data to prevent look-ahead
# - Designed to work in ranging markets (chop > 61.8) which occur in all regimes