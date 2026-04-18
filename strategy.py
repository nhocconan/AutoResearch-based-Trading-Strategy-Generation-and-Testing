#!/usr/bin/env python3
"""
12h_KAMA_Trend_With_1d_Trend_Filter
Hypothesis: KAMA (Kaufman Adaptive Moving Average) adapts to market noise, providing a smooth trend line that reduces whipsaws in ranging markets. Combined with a 1-day EMA trend filter and volume confirmation, this strategy aims to capture medium-term trends while avoiding false breakouts. Designed for 12-hour timeframe with ~15-30 trades/year to minimize fee drag and work in both bull and bear markets via adaptive trend filtering.
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
    
    # KAMA parameters
    er_length = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close, n=er_length))
    volatility = np.sum(np.abs(np.diff(close)), axis=0)
    # Handle edge case for volatility calculation
    volatility_padded = np.concatenate([np.zeros(er_length-1), volatility])
    er = np.where(volatility_padded != 0, change / volatility_padded, 0)
    er = np.concatenate([np.full(er_length-1, np.nan), er])
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.full_like(close, np.nan)
    kama[er_length-1] = close[er_length-1]  # Seed
    
    for i in range(er_length, n):
        if not np.isnan(sc[i]) and not np.isnan(kama[i-1]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
    
    # 1-day EMA trend filter (34-period)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: >1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Warmup for KAMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(kama[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(volume_filter[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        kama_val = kama[i]
        ema_trend = ema_1d_aligned[i]
        vol_ok = volume_filter[i]
        
        if position == 0:
            # Long: price above KAMA and EMA1d with volume
            if price > kama_val and price > ema_trend and vol_ok:
                signals[i] = 0.25
                position = 1
            # Short: price below KAMA and EMA1d with volume
            elif price < kama_val and price < ema_trend and vol_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long if price crosses below KAMA or trend fails
            if price < kama_val or price < ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short if price crosses above KAMA or trend fails
            if price > kama_val or price > ema_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Trend_With_1d_Trend_Filter"
timeframe = "12h"
leverage = 1.0