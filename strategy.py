#!/usr/bin/env python3
"""
4h_4hKAMA_Direction_1dRSI_Pullback_Volume_v3
Hypothesis: Trade KAMA direction on 4h with 1d RSI pullback and volume confirmation. KAMA adapts to market efficiency, reducing whipsaw in sideways markets and capturing trends. Enter long when KAMA turns up and 1d RSI < 35 (pullback in uptrend), short when KAMA turns down and 1d RSI > 65 (pullback in downtrend). Use volume > 2x 24-period average for confirmation. This version tightens conditions (RSI thresholds and volume multiplier) to reduce trade frequency and avoid overtrading. Works in bull/bear by following adaptive trend. Uses 1d RSI for pullback to avoid chasing extended moves.
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
    
    # Get 1d data for RSI
    df_1d = get_htf_data(prices, '1d')
    
    # 1d RSI(14)
    rsi_period = 14
    close_1d = df_1d['close'].values
    rsi_1d = np.full_like(close_1d, np.nan)
    
    if len(close_1d) >= rsi_period + 1:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(close_1d, np.nan)
        avg_loss = np.full_like(close_1d, np.nan)
        
        # First average
        avg_gain[rsi_period] = np.mean(gain[:rsi_period])
        avg_loss[rsi_period] = np.mean(loss[:rsi_period])
        
        # Wilder smoothing
        for i in range(rsi_period + 1, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * (rsi_period - 1) + gain[i-1]) / rsi_period
            avg_loss[i] = (avg_loss[i-1] * (rsi_period - 1) + loss[i-1]) / rsi_period
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_1d = 100 - (100 / (1 + rs))
    
    # Align 1d RSI to 4h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # KAMA on 4h
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # KAMA parameters
    fast_sc = 2 / (2 + 1)  # 2-period EMA
    slow_sc = 2 / (30 + 1) # 30-period EMA
    
    kama = np.full_like(close_4h, np.nan)
    
    if len(close_4h) >= 2:
        kama[0] = close_4h[0]
        for i in range(1, len(close_4h)):
            # Efficiency ratio
            if i >= 1:
                change = abs(close_4h[i] - close_4h[i-1])
                volatility = 0
                for j in range(1, i+1):
                    volatility += abs(close_4h[j] - close_4h[j-1])
                er = change / volatility if volatility != 0 else 0
                sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
                kama[i] = kama[i-1] + sc * (close_4h[i] - kama[i-1])
            else:
                kama[i] = close_4h[i]
    
    # Align KAMA to 4h timeframe (same as input, but using alignment for consistency)
    kama_aligned = align_htf_to_ltf(prices, df_4h, kama)
    
    # Volume confirmation: volume > 2x 24-period average (tightened)
    vol_ma = np.full_like(volume, np.nan)
    vol_period = 24
    
    if len(volume) >= vol_period:
        for i in range(vol_period, len(volume)):
            vol_ma[i] = np.mean(volume[i - vol_period:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, vol_period)  # KAMA needs ~30 periods, vol MA needs 24
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume[i] > 2.0 * vol_ma[i]  # Tightened from 1.5x to 2.0x
        
        if position == 0:
            # Long: KAMA turning up + RSI pullback (<35) + volume
            if i > 0 and not np.isnan(kama_aligned[i-1]) and kama_aligned[i] > kama_aligned[i-1] and rsi_1d_aligned[i] < 35 and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: KAMA turning down + RSI pullback (>65) + volume
            elif i > 0 and not np.isnan(kama_aligned[i-1]) and kama_aligned[i] < kama_aligned[i-1] and rsi_1d_aligned[i] > 65 and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down or RSI > 65 (overbought)
            if (i > 0 and not np.isnan(kama_aligned[i-1]) and kama_aligned[i] < kama_aligned[i-1]) or rsi_1d_aligned[i] > 65:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up or RSI < 35 (oversold)
            if (i > 0 and not np.isnan(kama_aligned[i-1]) and kama_aligned[i] > kama_aligned[i-1]) or rsi_1d_aligned[i] < 35:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_4hKAMA_Direction_1dRSI_Pullback_Volume_v3"
timeframe = "4h"
leverage = 1.0