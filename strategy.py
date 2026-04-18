#!/usr/bin/env python3
"""
1d_KAMA_Direction_WeeklyTrend_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) on daily timeframe adapts to market noise,
providing trend direction that avoids whipsaws in ranging markets. Combined with weekly trend
filter (EMA34 on weekly) and volume confirmation, it captures strong trends while avoiding
false signals in chop. Works in bull markets by riding uptrends and in bear markets by
avoiding false longs and taking shorts when weekly trend turns down.
Target: 15-25 trades/year (60-100 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 35:
        return np.zeros(n)
    
    # === KAMA on Daily (ER=10, FAST=2, SLOW=30) ===
    def kama(close_vals, er_period=10, fast=2, slow=30):
        n = len(close_vals)
        change = np.abs(np.diff(close_vals, k=er_period))  # |close[t] - close[t-er]|
        volatility = np.sum(np.abs(np.diff(close_vals)), axis=0) if False else None  # placeholder
        
        # Calculate ER (Efficiency Ratio) properly
        change = np.abs(np.diff(close_vals, k=er_period))
        # Volatility = sum of absolute changes over er_period window
        volatility = np.zeros_like(close_vals)
        for i in range(er_period, len(close_vals)):
            volatility[i] = np.sum(np.abs(np.diff(close_vals[i-er_period+1:i+1])))
        
        # Avoid division by zero
        er = np.zeros_like(close_vals)
        mask = volatility != 0
        er[mask] = change[mask] / volatility[mask]
        
        # Smooth constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # Calculate KAMA
        kama_vals = np.zeros_like(close_vals)
        kama_vals[0] = close_vals[0]
        for i in range(1, n):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close_vals[i] - kama_vals[i-1])
        
        return kama_vals
    
    # Calculate KAMA on daily close
    kama_1d = kama(df_1d['close'].values)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # === Weekly EMA34 for trend filter ===
    ema_1w = pd.Series(df_1w['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    # === Volume filter: >1.3x 20-period average ===
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 30  # Warmup for indicators
    
    for i in range(start_idx, n):
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(ema_1w_aligned[i]) or
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama_1d_aligned[i]
        ema_trend = ema_1w_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price above KAMA and above weekly EMA (uptrend) with volume
            if price > kama_val and price > ema_trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and below weekly EMA (downtrend) with volume
            elif price < kama_val and price < ema_trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long: exit if price crosses below KAMA or weekly EMA turns down
            if price < kama_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short: exit if price crosses above KAMA or weekly EMA turns up
            if price > kama_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_WeeklyTrend_Filter"
timeframe = "1d"
leverage = 1.0