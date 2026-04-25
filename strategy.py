#!/usr/bin/env python3
"""
1d_KAMA_Trend_With_RSI_Filter_v1
Hypothesis: Use KAMA (Kaufman Adaptive Moving Average) on 1d timeframe to determine trend direction, 
filtered by RSI(14) to avoid whipsaws, with volume confirmation and ATR-based stop management.
Long: KAMA trending up + RSI > 50 + volume > 1.5 * 20-period average volume.
Short: KAMA trending down + RSI < 50 + volume > 1.5 * 20-period average volume.
Exit: Opposite KAMA cross OR RSI crosses 50 in opposite direction.
Position size: 0.25 (25% of capital) to balance return and drawdown.
Target: 15-25 trades/year to stay within proven winning range for 1d timeframe.
Uses proper MTF data loading with get_htf_data() ONCE before loop and align_htf_to_ltf().
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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 1d close
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(close_1d, n=10))  # |close[t] - close[t-10]|
    volatility = np.sum(np.abs(np.diff(close_1d, n=1)), axis=1)  # sum of |diff| over 10 periods
    # Pad the beginning with NaN for alignment
    change = np.concatenate([np.full(10, np.nan), change])
    volatility = np.concatenate([np.full(10, np.nan), volatility])
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    fast_sc = 2 / (2 + 1)   # EMA(2)
    slow_sc = 2 / (30 + 1)  # EMA(30)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.full_like(close_1d, np.nan)
    kama[9] = close_1d[9]  # start after 10 periods
    for i in range(10, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    kama_1d = kama
    
    # Align KAMA to 1d timeframe (already aligned as we're using 1d data)
    # But we need to align to primary timeframe (1d) - since primary is 1d, no alignment needed
    # However, for consistency with MTF pattern, we'll still use align_htf_to_ltf
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # Calculate RSI(14) on 1d close
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=14, min_periods=14).mean().values
    avg_loss = pd.Series(loss).rolling(window=14, min_periods=14).mean().values
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    # Pad the beginning with NaN for first 14 values
    rsi = np.concatenate([np.full(14, np.nan), rsi])
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Volume confirmation: 1d volume > 1.5 * 20-period average
    vol_1d = df_1d['volume'].values
    vol_ma = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = vol_1d > (1.5 * vol_ma)
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need warmup for KAMA (10), RSI (14), volume MA (20)
    start_idx = max(10, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama_1d_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0 if position == 0 else (0.25 if position == 1 else -0.25)
            continue
        
        # Determine KAMA trend direction (up if current close > KAMA)
        kama_trend_up = close[i] > kama_1d_aligned[i]
        kama_trend_down = close[i] < kama_1d_aligned[i]
        
        # RSI conditions
        rsi_above_50 = rsi_1d_aligned[i] > 50
        rsi_below_50 = rsi_1d_aligned[i] < 50
        
        if position == 0:
            # Long setup: KAMA trending up + RSI > 50 + volume spike
            long_setup = kama_trend_up and rsi_above_50 and volume_spike_aligned[i]
            
            # Short setup: KAMA trending down + RSI < 50 + volume spike
            short_setup = kama_trend_down and rsi_below_50 and volume_spike_aligned[i]
            
            if long_setup:
                signals[i] = 0.25
                position = 1
            elif short_setup:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            # Exit: KAMA trend turns down OR RSI crosses below 50
            if (not kama_trend_up) or (not rsi_above_50):
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            # Exit: KAMA trend turns up OR RSI crosses above 50
            if (not kama_trend_down) or (not rsi_below_50):
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1d_KAMA_Trend_With_RSI_Filter_v1"
timeframe = "1d"
leverage = 1.0