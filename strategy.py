#!/usr/bin/env python3
"""
1d_1w_KAMA_Trend_Filter_v2
Hypothesis: Uses weekly KAMA (Kaufman Adaptive Moving Average) to determine trend direction,
with daily price crossing above/below KAMA as entry signal, filtered by volume spike and
ATR-based volatility filter. Designed to work in both bull and bear markets by following
higher-timeframe trend while using daily timeframe for precise entries. Targets low trade
frequency (7-25/year) via weekly trend filter and daily entry signal.
"""

name = "1d_1w_KAMA_Trend_Filter_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, period=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average"""
    change = abs(np.diff(close, n=period))
    volatility = np.abs(np.diff(close)).rolling(window=period, min_periods=1).sum()
    er = np.where(volatility != 0, change / volatility, 0)
    sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
    kama = np.zeros_like(close)
    kama[0] = close[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # 1d OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # --- Weekly KAMA for Trend Filter ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    kama_1w = calculate_kama(df_1w['close'].values, period=10, fast=2, slow=30)
    kama_1w_1d = align_htf_to_ltf(prices, df_1w, kama_1w)
    
    # --- Daily ATR for Volatility Filter ---
    tr1 = high[1:] - low[1:]
    tr2 = np.abs(high[1:] - close[:-1])
    tr3 = np.abs(low[1:] - close[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # --- Daily Volume Spike Detection (20-period average) ---
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    vol_ratio = np.nan_to_num(vol_ratio, nan=1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1w_1d[i]) or np.isnan(atr[i]) or np.isnan(vol_ratio[i])):
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Trend filter: price relative to weekly KAMA
        price_above_kama = close[i] > kama_1w_1d[i]
        price_below_kama = close[i] < kama_1w_1d[i]
        
        # Volume confirmation threshold
        volume_spike = vol_ratio[i] > 1.5
        
        # Volatility filter: avoid extremely high volatility periods
        vol_filter = atr[i] < np.percentile(atr[:i+1], 80) if i >= 20 else True
        
        if position == 0:
            # Long: price above weekly KAMA + volume spike + volatility filter
            if price_above_kama and volume_spike and vol_filter:
                signals[i] = 0.25
                position = 1
            # Short: price below weekly KAMA + volume spike + volatility filter
            elif price_below_kama and volume_spike and vol_filter:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: price crosses back across weekly KAMA
            if position == 1:
                # Exit long: price crosses below weekly KAMA
                if price_below_kama:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Exit short: price crosses above weekly KAMA
                if price_above_kama:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals