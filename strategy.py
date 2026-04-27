#!/usr/bin/env python3
"""
4h_KAMA_Direction_RSI_Filter_Volume
Hypothesis: KAMA (Kaufman Adaptive Moving Average) filters noise and adapts to market regime. In trending markets, KAMA follows price closely; in ranging markets, it flattens. Combined with RSI for momentum confirmation and volume filter to avoid false signals, this strategy captures sustained moves while minimizing whipsaws. Works in both bull and bear markets by adapting to volatility. Targets ~25 trades/year on 4h to minimize fee drag.
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
    
    # Get 4h data for KAMA calculation (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on 4h data
    close_4h = df_4h['close'].values
    # Efficiency Ratio: ER = |close - close[10]| / sum(|close - close[1]|) over 10 periods
    change = np.abs(close_4h - np.roll(close_4h, 10))
    volatility = np.sum(np.abs(np.diff(close_4h, axis=0)), axis=0) if len(close_4h) > 1 else np.zeros_like(close_4h)
    # Vectorized volatility sum over 10 periods
    vol_sum = np.zeros_like(close_4h)
    for i in range(10, len(close_4h)):
        vol_sum[i] = np.sum(np.abs(np.diff(close_4h[i-9:i+1])))
    er = np.where(vol_sum > 0, change / vol_sum, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1))**2  # fast=2, slow=30
    # KAMA calculation
    kama = np.zeros_like(close_4h)
    kama[0] = close_4h[0]
    for i in range(1, len(close_4h)):
        kama[i] = kama[i-1] + sc[i] * (close_4h[i] - kama[i-1])
    
    # Align KAMA to 15m timeframe (if needed) but we're using 4h as primary
    # Since we're using 4h data directly, no alignment needed for same timeframe
    # But we need to map 4h indices to 15m indices: each 4h bar = 16x 15m bars
    # We'll create a mapping from 4h index to 15m index
    kama_4h = kama
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_spike = volume > (vol_ma * 2.0)
    
    # RSI(14) for momentum
    def calculate_rsi(prices, period=14):
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(prices)
        avg_loss = np.zeros_like(prices)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(prices)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi = calculate_rsi(close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # Position size: 25% of capital
    
    # Warmup: need enough data for KAMA, EMA, RSI, and volume MA
    start_idx = max(50, 30, 20)
    
    for i in range(start_idx, n):
        # Map 15m index to 4h index for KAMA
        # Each 4h bar = 16x 15m bars (since 4h = 240min, 15m = 15min, 240/15=16)
        kama_idx = i // 16
        if kama_idx >= len(kama_4h):
            kama_idx = len(kama_4h) - 1
        
        # Skip if any data not ready
        if (np.isnan(kama_4h[kama_idx]) or 
            np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(rsi[i])):
            signals[i] = 0.0
            continue
        
        kama_val = kama_4h[kama_idx]
        ema_trend = ema50_1d_aligned[i]
        rsi_val = rsi[i]
        vol_spike_val = vol_spike[i]
        
        if position == 0:
            # Long: price above KAMA (uptrend), RSI > 50, volume spike
            if close[i] > kama_val and rsi_val > 50 and vol_spike_val and close[i] > ema_trend:
                signals[i] = size
                position = 1
            # Short: price below KAMA (downtrend), RSI < 50, volume spike
            elif close[i] < kama_val and rsi_val < 50 and vol_spike_val and close[i] < ema_trend:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price falls below KAMA or RSI < 40
            if close[i] < kama_val or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price rises above KAMA or RSI > 60
            if close[i] > kama_val or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_KAMA_Direction_RSI_Filter_Volume"
timeframe = "4h"
leverage = 1.0