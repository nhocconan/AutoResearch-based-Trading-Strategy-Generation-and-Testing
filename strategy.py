#!/usr/bin/env python3
"""
12h_1d_kama_rsi_chop_filter_v1
Hypothesis: 12-hour strategy using KAMA trend direction, RSI for momentum, and Choppiness index for regime filter.
KAMA adapts to market noise, RSI avoids overbought/oversold extremes, and Choppiness filters for trending vs ranging markets.
Works in bull/bear by requiring KAMA alignment and avoiding counter-trend entries in strong trends.
Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
"""

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
    
    # Get 1d data for Choppiness index (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # KAMA on 12h for trend direction
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate Efficiency Ratio and Smoothing Constants for KAMA
    change_12h = np.abs(np.diff(close_12h, prepend=close_12h[0]))
    volatility_12h = np.sum(np.abs(np.diff(close_12h, prepend=close_12h[0])), axis=0) if False else None  # placeholder
    
    # Proper ER calculation: need rolling sum of absolute changes
    change_12h_series = pd.Series(close_12h)
    volatility_12h_series = change_12h_series.diff().abs().rolling(window=10, min_periods=10).sum()
    direction_12h = change_12h_series.diff(periods=10).abs()
    er_12h = np.where(volatility_12h_series > 0, direction_12h / volatility_12h_series, 0)
    sc_12h = (er_12h * 0.064 + 0.062) ** 2  # where 0.064 = 2/(2+1), 0.062 = 2/(30+1)
    
    # Calculate KAMA
    kama_12h = np.full_like(close_12h, np.nan)
    kama_12h[9] = close_12h[9]  # seed
    for i in range(10, len(close_12h)):
        kama_12h[i] = kama_12h[i-1] + sc_12h[i] * (close_12h[i] - kama_12h[i-1])
    
    kama_12h_aligned = align_htf_to_ltf(prices, df_12h, kama_12h)
    
    # RSI on 12h for momentum (avoid extremes)
    rsi_period = 14
    delta = pd.Series(close_12h).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
    avg_loss = loss.ewm(alpha=1/rsi_period, min_periods=rsi_period).mean()
    rs = avg_gain / avg_loss
    rsi_12h = 100 - (100 / (1 + rs))
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h.values)
    
    # Choppiness Index on 1d (HTF) for regime filter
    # True Range
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]  # first period
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) - sum of TR over 14 periods
    atr_1d = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    hh_1d = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll_1d = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Chop = 100 * log10( sum(tr1..tr14) / (HH - LL) ) / log10(14)
    chop_1d = 100 * np.log10(atr_1d / (hh_1d - ll_1d)) / np.log10(14)
    chop_1d = np.where((hh_1d - ll_1d) > 0, chop_1d, 50)  # avoid division by zero
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume confirmation on 12h
    vol_12h = df_12h['volume'].values
    vol_ma_12h = pd.Series(vol_12h).rolling(window=20, min_periods=20).mean().values
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    vol_confirm = volume > (vol_ma_12h_aligned * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(kama_12h_aligned[i]) or np.isnan(rsi_12h_aligned[i]) or 
            np.isnan(chop_1d_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long conditions: price above KAMA (uptrend), RSI not overbought, Chop < 61.8 (trending)
        if (close[i] > kama_12h_aligned[i] and 
            rsi_12h_aligned[i] < 70 and 
            chop_1d_aligned[i] < 61.8 and 
            vol_confirm[i] and 
            position != 1):
            position = 1
            signals[i] = 0.25
        # Short conditions: price below KAMA (downtrend), RSI not oversold, Chop < 61.8 (trending)
        elif (close[i] < kama_12h_aligned[i] and 
              rsi_12h_aligned[i] > 30 and 
              chop_1d_aligned[i] < 61.8 and 
              vol_confirm[i] and 
              position != -1):
            position = -1
            signals[i] = -0.25
        # Exit: reverse signal or Chop > 61.8 (ranging market)
        elif position == 1 and (close[i] < kama_12h_aligned[i] or chop_1d_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        elif position == -1 and (close[i] > kama_12h_aligned[i] or chop_1d_aligned[i] > 61.8):
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_kama_rsi_chop_filter_v1"
timeframe = "12h"
leverage = 1.0