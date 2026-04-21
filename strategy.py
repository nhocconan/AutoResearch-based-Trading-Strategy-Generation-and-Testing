#!/usr/bin/env python3
"""
Hypothesis: 6h KAMA trend with daily RSI momentum and volume confirmation.
Long when KAMA rising (bullish trend) and daily RSI > 50 (bullish momentum) with volume > 1.5x average.
Short when KAMA falling (bearish trend) and daily RSI < 50 (bearish momentum) with volume > 1.5x average.
Exit when KAMA direction reverses or RSI crosses 50.
KAMA adapts to market noise, reducing whipsaw in chop. RSI filters momentum direction.
Designed for 15-25 trades/year to minimize fee drift while capturing sustained trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load daily data ONCE before loop for KAMA and RSI calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily price arrays
    close_1d = df_1d['close'].values
    vol_1d = df_1d['volume'].values
    
    # KAMA (Kaufman Adaptive Moving Average) parameters
    er_period = 10
    fast_sc = 2 / (2 + 1)  # EMA(2)
    slow_sc = 2 / (30 + 1) # EMA(30)
    
    # Calculate Efficiency Ratio (ER)
    change = np.abs(np.diff(close_1d, n=er_period))
    volatility = np.sum(np.abs(np.diff(close_1d)), axis=0)
    # Handle volatility calculation for array
    volatility_full = np.zeros_like(close_1d)
    for i in range(er_period, len(close_1d)):
        volatility_full[i] = np.sum(np.abs(np.diff(close_1d[i-er_period:i])))
    er = np.zeros_like(close_1d)
    er[er_period:] = change / np.where(volatility_full[er_period:] == 0, 1, volatility_full[er_period:])
    
    # Smoothing constant
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA calculation
    kama = np.zeros_like(close_1d)
    kama[0] = close_1d[0]
    for i in range(1, len(close_1d)):
        kama[i] = kama[i-1] + sc[i] * (close_1d[i] - kama[i-1])
    
    # RSI (14-period)
    delta = np.diff(close_1d)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(close_1d)
    avg_loss = np.zeros_like(close_1d)
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(close_1d)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss == 0, 100, avg_gain / avg_loss)
    rsi = 100 - (100 / (1 + rs))
    
    # Daily volume average (20-period)
    vol_ma_20 = np.zeros_like(vol_1d)
    for i in range(20, len(vol_1d)):
        vol_ma_20[i] = np.mean(vol_1d[i-20:i])
    
    # Align all indicators to 6h timeframe
    kama_aligned = align_htf_to_ltf(prices, df_1d, kama)
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    vol_ma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_20)
    vol_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(kama_aligned[i]) or np.isnan(rsi_aligned[i]) or 
            np.isnan(vol_ma_20_aligned[i]) or np.isnan(vol_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Current daily values aligned to 6h
        vol_1d_current = vol_1d_aligned[i]
        
        if position == 0:
            # Enter long: KAMA rising, RSI > 50, volume surge
            if (kama_aligned[i] > kama_aligned[max(i-1, 0)] and 
                rsi_aligned[i] > 50 and
                vol_1d_current > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: KAMA falling, RSI < 50, volume surge
            elif (kama_aligned[i] < kama_aligned[max(i-1, 0)] and 
                  rsi_aligned[i] < 50 and
                  vol_1d_current > 1.5 * vol_ma_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit: KAMA direction reverses or RSI crosses 50
            exit_signal = False
            
            if position == 1:
                # Exit long: KAMA falling OR RSI <= 50
                if (kama_aligned[i] <= kama_aligned[max(i-1, 0)] or 
                    rsi_aligned[i] <= 50):
                    exit_signal = True
            elif position == -1:
                # Exit short: KAMA rising OR RSI >= 50
                if (kama_aligned[i] >= kama_aligned[max(i-1, 0)] or 
                    rsi_aligned[i] >= 50):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_KAMA_RSI_Momentum_Volume1.5x"
timeframe = "6h"
leverage = 1.0