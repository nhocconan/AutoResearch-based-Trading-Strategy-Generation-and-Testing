#!/usr/bin/env python3
"""
12h_Volume_Weighted_SMA_Crossover_RSI_Filter
Hypothesis: Combines volume-weighted SMA crossovers (9/21) on 12h chart with RSI momentum filter and volume confirmation.
Designed to capture medium-term trends while avoiding whipsaws in low-volume or ranging conditions.
Volume-weighted SMAs give more significance to price action on high-volume bars, improving signal quality.
RSI filter prevents entries during overextended moves. Target: 15-25 trades/year per symbol.
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
    
    # Get 12h data for indicators
    df_12h = get_htf_data(prices, '12h')
    
    # Calculate volume-weighted SMA (9 and 21) on 12h close
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    def vwma(source, length):
        """Volume Weighted Moving Average"""
        result = np.full_like(source, np.nan)
        if len(source) < length:
            return result
        for i in range(length-1, len(source)):
            window_close = source[i-length+1:i+1]
            window_vol = volume_12h[i-length+1:i+1]
            vol_sum = np.sum(window_vol)
            if vol_sum > 0:
                result[i] = np.sum(window_close * window_vol) / vol_sum
            else:
                result[i] = np.nan
        return result
    
    vwma9 = vwma(close_12h, 9)
    vwma21 = vwma(close_12h, 21)
    
    # Calculate RSI(14) on 12h close
    def rsi(source, length=14):
        delta = np.diff(source, prepend=source[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full_like(source, np.nan)
        avg_loss = np.full_like(source, np.nan)
        
        # Wilder's smoothing
        if len(source) >= length:
            avg_gain[length-1] = np.mean(gain[1:length+1])
            avg_loss[length-1] = np.mean(loss[1:length+1])
            
            for i in range(length, len(source)):
                avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
                avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        
        rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
        rsi_val = 100 - (100 / (1 + rs))
        return rsi_val
    
    rsi_12h = rsi(close_12h, 14)
    
    # Volume confirmation: current 12h volume > 1.5 x 20-period average
    vol_ma_12h = np.full_like(volume_12h, np.nan)
    for i in range(20, len(volume_12h)):
        vol_ma_12h[i] = np.mean(volume_12h[i-20:i])
    vol_confirm_12h = volume_12h > (vol_ma_12h * 1.5)
    
    # Align all indicators to 12h timeframe (primary)
    vwma9_aligned = align_htf_to_ltf(prices, df_12h, vwma9)
    vwma21_aligned = align_htf_to_ltf(prices, df_12h, vwma21)
    rsi_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi_12h)
    vol_confirm_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_confirm_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(21, 20)  # Need enough data for VWMA21 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(vwma9_aligned[i]) or np.isnan(vwma21_aligned[i]) or 
            np.isnan(rsi_12h_aligned[i]) or np.isnan(vol_confirm_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: VWMA9 crosses above VWMA21, RSI not overbought, volume confirmation
            if (vwma9_aligned[i] > vwma21_aligned[i] and 
                rsi_12h_aligned[i] < 70 and 
                vol_confirm_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: VWMA9 crosses below VWMA21, RSI not oversold, volume confirmation
            elif (vwma9_aligned[i] < vwma21_aligned[i] and 
                  rsi_12h_aligned[i] > 30 and 
                  vol_confirm_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: VWMA9 crosses below VWMA21 or RSI overbought
            if (vwma9_aligned[i] < vwma21_aligned[i] or 
                rsi_12h_aligned[i] > 75):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: VWMA9 crosses above VWMA21 or RSI oversold
            if (vwma9_aligned[i] > vwma21_aligned[i] or 
                rsi_12h_aligned[i] < 25):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Volume_Weighted_SMA_Crossover_RSI_Filter"
timeframe = "12h"
leverage = 1.0