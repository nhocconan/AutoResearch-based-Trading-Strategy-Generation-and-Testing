#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Filter
KAMA-based trend following with RSI filter for trend strength.
- Long when KAMA is rising and RSI > 50 (bullish momentum)
- Short when KAMA is falling and RSI < 50 (bearish momentum)
- Exit when KAMA direction changes or RSI crosses 50
- Uses 1w trend filter to avoid counter-trend trades in strong trends
- Designed for 10-20 trades/year per symbol
Works in both bull (captures trends) and bear (avoids false signals) markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_kama(close, er_length=10, fast=2, slow=30):
    """Calculate Kaufman Adaptive Moving Average."""
    n = len(close)
    kama = np.full(n, np.nan)
    if n == 0:
        return kama
    
    # Efficiency Ratio
    change = np.abs(np.diff(close, prepend=close[0]))
    abs_change = np.abs(np.diff(close, prepend=close[0]))
    
    # For efficiency ratio, we need rolling sum
    er = np.zeros(n)
    for i in range(er_length, n):
        if i >= er_length:
            net_change = abs(close[i] - close[i-er_length])
            total_change = np.sum(abs_change[i-er_length+1:i+1])
            if total_change > 0:
                er[i] = net_change / total_change
            else:
                er[i] = 0
    
    # Smoothing constants
    sc = np.zeros(n)
    for i in range(n):
        fast_sc = 2.0 / (fast + 1)
        slow_sc = 2.0 / (slow + 1)
        sc[i] = (er[i] * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama[0] = close[0]
    for i in range(1, n):
        if not np.isnan(sc[i]):
            kama[i] = kama[i-1] + sc[i] * (close[i] - kama[i-1])
        else:
            kama[i] = kama[i-1]
    
    return kama

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    
    # Get 1d data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d KAMA
    kama_1d = calculate_kama(close_1d, er_length=10, fast=2, slow=30)
    
    # Align 1d KAMA to 1d timeframe (no alignment needed since we're on 1d)
    kama_1d_aligned = kama_1d  # Already on 1d timeframe
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate 1w EMA for trend filter
    if len(close_1w) >= 50:
        ema_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
        ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    else:
        ema_1w_aligned = np.full(n, np.nan)
    
    # Calculate RSI on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    rsi = np.zeros_like(close_1d)
    
    # Wilder's smoothing
    for i in range(1, len(close_1d)):
        if i < 14:
            avg_gain[i] = np.mean(gain[1:i+1]) if i > 0 else 0
            avg_loss[i] = np.mean(loss[1:i+1]) if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_aligned = rsi  # Already on 1d timeframe
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need sufficient data for calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(ema_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction (1 = rising, -1 = falling, 0 = flat)
        if i > 0 and not np.isnan(kama_1d_aligned[i-1]):
            if kama_1d_aligned[i] > kama_1d_aligned[i-1]:
                kama_dir = 1
            elif kama_1d_aligned[i] < kama_1d_aligned[i-1]:
                kama_dir = -1
            else:
                kama_dir = 0
        else:
            kama_dir = 0
        
        # Trend filter: price above/below 1w EMA
        price_above_1w_ema = close[i] > ema_1w_aligned[i]
        price_below_1w_ema = close[i] < ema_1w_aligned[i]
        
        if position == 0:
            # Long: KAMA rising + RSI > 50 + price above 1w EMA (bullish alignment)
            if kama_dir == 1 and rsi_aligned[i] > 50 and price_above_1w_ema:
                signals[i] = 0.25
                position = 1
            # Short: KAMA falling + RSI < 50 + price below 1w EMA (bearish alignment)
            elif kama_dir == -1 and rsi_aligned[i] < 50 and price_below_1w_ema:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA falling OR RSI < 50
            if kama_dir == -1 or rsi_aligned[i] < 50:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA rising OR RSI > 50
            if kama_dir == 1 or rsi_aligned[i] > 50:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Filter"
timeframe = "1d"
leverage = 1.0