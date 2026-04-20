#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_KAMA_Trend_With_Range_Filter_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Get daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === Daily Close for Trend ===
    close_1d = df_1d['close'].values
    
    # KAMA: Kaufman Adaptive Moving Average (10-period ER)
    close_1d_series = pd.Series(close_1d)
    change = np.abs(close_1d_series.diff(10))
    volatility = close_1d_series.diff().abs().rolling(10).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (0.6645 - 0.0645) + 0.0645) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # === 4h Price ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # 4h ATR for volatility filter
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = np.zeros_like(close)
    atr[0] = np.mean(tr[:20]) if len(tr) >= 20 else np.mean(tr) if len(tr) > 0 else 0
    for i in range(1, len(tr)):
        atr[i+1] = (atr[i] * 13 + tr[i]) / 14
    
    # Align daily KAMA to 4h
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # === 4h Bollinger Bands for Range Detection ===
    close_series = pd.Series(close)
    basis = close_series.rolling(window=20, min_periods=20).mean().values
    dev = close_series.rolling(window=20, min_periods=20).std().values
    upper = basis + 2.0 * dev
    lower = basis - 2.0 * dev
    bb_width = (upper - lower) / np.where(basis > 0, basis, np.nan)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Get values
        close_val = close[i]
        kama_val = kama_aligned[i]
        bb_width_val = bb_width[i]
        
        # Skip if any value is NaN
        if (np.isnan(close_val) or np.isnan(kama_val) or np.isnan(bb_width_val)):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Range filter: only trade in low volatility (BB width < 0.05 = 5%)
        if bb_width_val >= 0.05:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price above KAMA (uptrend) in low volatility
            if close_val > kama_val:
                signals[i] = 0.25
                position = 1
            # Short: Price below KAMA (downtrend) in low volatility
            elif close_val < kama_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: Price crosses below KAMA
            if close_val < kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: Price crosses above KAMA
            if close_val > kama_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals