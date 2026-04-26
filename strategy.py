#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_Volume_Filter
Hypothesis: Kaufman Adaptive Moving Average (KAMA) captures trend with lower lag and fewer whipsaws than traditional MA. Combined with volume confirmation (>1.5x 20-bar average) and 1-week EMA50 trend filter, this strategy aims for high-probability entries in both bull and bear markets. The 1d timeframe reduces trade frequency to minimize fee drag, while KAMA's adaptive nature improves trend detection during volatile periods. Target: 15-25 trades/year (60-100 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data ONCE before loop for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Load 1w data ONCE before loop for HTF trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close
    close_1d = df_1d['close'].values
    kama = calculate_kama(close_1d, er_len=10, fast_ma=2, slow_ma=30)
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    
    # 1w EMA50 for trend filter
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    base_size = 0.25  # Position size (25% of capital)
    
    # Warmup: max of calculations (20 for vol, 30 for KAMA, 50 for 1w EMA)
    start_idx = max(20, 30, 50)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = base_size if position == 1 else (-base_size if position == -1 else 0.0)
            continue
        
        close_val = close[i]
        kama_val = kama_aligned[i]
        ema_50_val = ema_50_1w_aligned[i]
        vol_spike = volume_spike[i]
        
        # Determine 1w trend: bullish if price > EMA50, bearish if price < EMA50
        bullish_1w = close_val > ema_50_val
        bearish_1w = close_val < ema_50_val
        
        # Entry conditions: price crosses KAMA in trend direction with volume spike
        long_entry = (close_val > kama_val) and bullish_1w and vol_spike
        short_entry = (close_val < kama_val) and bearish_1w and vol_spike
        
        if position == 0:
            # Flat - look for entry
            if long_entry:
                signals[i] = base_size
                position = 1
            elif short_entry:
                signals[i] = -base_size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long - exit on mean reversion below KAMA or trend change
            if close_val < kama_val or not bullish_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = base_size
        elif position == -1:
            # Short - exit on mean reversion above KAMA or trend change
            if close_val > kama_val or not bearish_1w:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -base_size
    
    return signals

def calculate_kama(close, er_len=10, fast_ma=2, slow_ma=30):
    """Calculate Kaufman Adaptive Moving Average"""
    close = pd.Series(close)
    # Efficiency Ratio
    change = abs(close.diff(er_len))
    volatility = close.diff().abs().rolling(er_len).sum()
    er = change / volatility.replace(0, 1e-10)
    # Smoothing Constants
    sc = (er * (2/(fast_ma+1) - 2/(slow_ma+1)) + 2/(slow_ma+1)) ** 2
    # KAMA
    kama = [np.nan] * len(close)
    kama[0] = close.iloc[0]
    for i in range(1, len(close)):
        kama[i] = kama[i-1] + sc.iloc[i] * (close.iloc[i] - kama[i-1])
    return np.array(kama)

name = "1d_KAMA_Trend_With_Volume_Filter"
timeframe = "1d"
leverage = 1.0