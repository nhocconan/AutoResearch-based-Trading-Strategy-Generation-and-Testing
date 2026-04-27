#!/usr/bin/env python3
"""
Hypothesis: Daily KAMA trend with RSI momentum filter and volume confirmation.
Enters long when KAMA turns up, RSI > 55, and volume > 1.5x 20-day average.
Enters short when KAMA turns down, RSI < 45, and volume > 1.5x 20-day average.
Uses weekly timeframe for trend confirmation (weekly KAMA direction).
Designed to capture momentum in trending markets while avoiding whipsaws in ranging conditions.
Target: 15-25 trades/year per symbol (60-100 total over 4 years) to minimize fee drag.
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
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate daily KAMA (adaptive moving average)
    close_1d = df_1d['close'].values
    # Efficiency ratio
    change = np.abs(np.diff(close_1d, prepend=close_1d[0]))
    volatility = np.abs(np.diff(close_1d))
    er = np.zeros_like(close_1d)
    for i in range(1, len(close_1d)):
        if np.sum(volatility[max(0, i-9):i+1]) > 0:
            er[i] = np.sum(change[max(0, i-9):i+1]) / np.sum(volatility[max(0, i-9):i+1])
        else:
            er[i] = 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    # Calculate KAMA
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Calculate daily RSI(14)
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    # Calculate daily volume average
    vol_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Get weekly data for trend confirmation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly KAMA for trend filter
    close_1w = df_1w['close'].values
    change_w = np.abs(np.diff(close_1w, prepend=close_1w[0]))
    volatility_w = np.abs(np.diff(close_1w))
    er_w = np.zeros_like(close_1w)
    for i in range(1, len(close_1w)):
        if np.sum(volatility_w[max(0, i-9):i+1]) > 0:
            er_w[i] = np.sum(change_w[max(0, i-9):i+1]) / np.sum(volatility_w[max(0, i-9):i+1])
        else:
            er_w[i] = 0
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc_w = (er_w * (fast_sc - slow_sc) + slow_sc) ** 2
    kama_w = np.zeros_like(close_1w)
    kama_w[0] = close_1w[0]
    for i in range(1, len(close_1w)):
        kama_w[i] = kama_w[i-1] + sc_w[i] * (close_1w[i] - kama_w[i-1])
    
    # Align daily indicators to lower timeframe (assumes 1h or similar)
    # For 1d timeframe, we can use values directly with proper indexing
    # Since we're using 1d timeframe, we need to align to 1d bars
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    kama_w_aligned = align_htf_to_ltf(prices, df_1w, kama_w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need enough data for calculations
    start_idx = 30
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(kama_w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current daily price and volume
        price_now = close[i]
        vol_now = volume[i]
        vol_ma = vol_ma_20_aligned[i]
        
        # Current indicators
        kama_now = kama_aligned[i]
        rsi_now = rsi_aligned[i]
        kama_w_now = kama_w_aligned[i]
        
        # Volume filter: volume > 1.5x 20-day average
        vol_filter = vol_now > 1.5 * vol_ma
        
        # KAMA direction: current vs previous
        kama_up = kama_now > kama_aligned[i-1]
        kama_down = kama_now < kama_aligned[i-1]
        
        # Entry conditions
        if position == 0:
            # Long: KAMA up, RSI > 55, volume confirmation, weekly uptrend
            if kama_up and rsi_now > 55 and vol_filter and price_now > kama_w_now:
                signals[i] = size
                position = 1
            # Short: KAMA down, RSI < 45, volume confirmation, weekly downtrend
            elif kama_down and rsi_now < 45 and vol_filter and price_now < kama_w_now:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: KAMA turns down or RSI < 50
            if not kama_up or rsi_now < 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: KAMA turns up or RSI > 50
            if not kama_down or rsi_now > 50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_KAMA_RSI_Volume_Trend"
timeframe = "1d"
leverage = 1.0