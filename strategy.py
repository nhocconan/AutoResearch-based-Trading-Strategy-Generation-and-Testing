#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h_1d_kama_rsi_volume_v1
# Uses Kaufman Adaptive Moving Average (KAMA) on 1d for trend direction,
# RSI(14) on 4h for momentum confirmation, and volume spike filter.
# Long when 1d KAMA rising and RSI < 70 (avoiding overbought),
# Short when 1d KAMA falling and RSI > 30 (avoiding oversold).
# Volume > 1.5x 20-period average confirms momentum.
# Designed for low trade frequency (<50/year) to minimize fee drag.
# Works in bull trends (KAMA up) and bear trends (KAMA down).

name = "4h_1d_kama_rsi_volume_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for KAMA calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate KAMA on daily close
    close_1d = df_1d['close'].values
    # Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, n=10))  # 10-period change
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)  # needs fixing
    # Correct ER calculation
    er = np.zeros_like(close_1d)
    for i in range(10, len(close_1d)):
        if i >= 10:
            ch = np.abs(close_1d[i] - close_1d[i-10])
            vol = np.sum(np.abs(np.diff(close_1d[i-10:i+1])))
            er[i] = ch / vol if vol != 0 else 0
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # Alternative simpler approach: use EMA as trend proxy if KAMA complex
    # But we'll implement proper KAMA
    
    # Recalculate ER properly
    change = np.abs(np.diff(close_1d, n=10))
    volatility = np.array([np.sum(np.abs(np.diff(close_1d[i-10:i+1]))) for i in range(10, len(close_1d))])
    er = np.zeros(len(close_1d))
    er[10:] = change / volatility
    er[volatility == 0] = 0
    
    # Smoothing constants
    fast_sc = 2 / (2 + 1)
    slow_sc = 2 / (30 + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # KAMA direction: rising if today > yesterday
    kama_rising = kama > np.roll(kama, 1)
    kama_falling = kama < np.roll(kama, 1)
    kama_rising[0] = False
    kama_falling[0] = False
    
    # Align KAMA to 4h
    kama_rising_aligned = align_htf_to_ltf(prices, df_1d, kama_rising)
    kama_falling_aligned = align_htf_to_ltf(prices, df_1d, kama_falling)
    
    # RSI on 4h
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        rs = avg_gain / (avg_loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    rsi_4h = rsi(close, 14)
    rsi_overbought = rsi_4h > 70
    rsi_oversold = rsi_4h < 30
    
    # Volume confirmation: volume > 1.5 * 20-period average
    vol_ma = np.zeros_like(volume)
    for i in range(20, len(volume)):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if np.isnan(kama_rising_aligned[i]) or np.isnan(kama_falling_aligned[i]) or np.isnan(rsi_4h[i]):
            signals[i] = 0.0
            continue
        
        # Require volume confirmation
        if not vol_confirm[i]:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
            continue
        
        # Long: KAMA rising AND RSI not overbought
        if kama_rising_aligned[i] and not rsi_overbought[i] and position != 1:
            position = 1
            signals[i] = 0.25
        # Short: KAMA falling AND RSI not oversold
        elif kama_falling_aligned[i] and not rsi_oversold[i] and position != -1:
            position = -1
            signals[i] = -0.25
        # Exit: opposite KAMA direction
        elif kama_falling_aligned[i] and position == 1:
            position = 0
            signals[i] = 0.0
        elif kama_rising_aligned[i] and position == -1:
            position = 0
            signals[i] = 0.0
        else:
            # Hold position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals