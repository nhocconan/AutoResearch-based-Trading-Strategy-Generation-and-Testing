#!/usr/bin/env python3
# 4h_KAMA_Trend_RSI_Extremes_ChopFilter_v2
# Hypothesis: KAMA trend direction combined with RSI extremes and Choppiness index regime filter.
# Works in bull/bear: KAMA adapts to market noise, RSI catches overextended moves, Choppiness filter avoids whipsaws in ranging markets.
# Uses 1d Choppiness index for regime detection (trending when < 38.2, ranging when > 61.8) and enters only in trending regimes.
# Position size: 0.25 for clear signals.

name = "4h_KAMA_Trend_RSI_Extremes_ChopFilter_v2"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate KAMA (Kaufman Adaptive Moving Average)
    def calculate_kama(close_prices, length=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close_prices, prepend=close_prices[0]))
        volatility = np.sum(np.abs(np.diff(close_prices)), axis=0)
        # Handle first element
        er = np.zeros_like(close_prices)
        er[0] = 0
        for i in range(1, len(close_prices)):
            if volatility[i] != 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        
        # Smoothing Constant
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        sc[0] = 0
        
        # KAMA
        kama = np.zeros_like(close_prices)
        kama[0] = close_prices[0]
        for i in range(1, len(close_prices)):
            kama[i] = kama[i-1] + sc[i] * (close_prices[i] - kama[i-1])
        return kama
    
    # Calculate RSI
    def calculate_rsi(close_prices, length=14):
        delta = np.diff(close_prices, prepend=close_prices[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close_prices)
        avg_loss = np.zeros_like(close_prices)
        
        # First average
        if len(close_prices) >= length:
            avg_gain[length-1] = np.mean(gain[0:length])
            avg_loss[length-1] = np.mean(loss[0:length])
        
        # Wilder smoothing
        for i in range(length, len(close_prices)):
            avg_gain[i] = (avg_gain[i-1] * (length-1) + gain[i]) / length
            avg_loss[i] = (avg_loss[i-1] * (length-1) + loss[i]) / length
        
        rs = np.zeros_like(close_prices)
        rs[avg_loss != 0] = avg_gain[avg_loss != 0] / avg_loss[avg_loss != 0]
        rsi = 100 - (100 / (1 + rs))
        rsi[avg_loss == 0] = 100
        return rsi
    
    # Calculate Choppiness Index
    def calculate_chop(high_prices, low_prices, close_prices, length=14):
        atr = np.zeros_like(close_prices)
        tr1 = np.abs(high_prices - low_prices)
        tr2 = np.abs(high_prices - np.roll(close_prices, 1))
        tr3 = np.abs(low_prices - np.roll(close_prices, 1))
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        tr[0] = tr1[0]  # First true range
        
        # ATR calculation
        atr[length-1] = np.mean(tr[0:length])
        for i in range(length, len(close_prices)):
            atr[i] = (atr[i-1] * (length-1) + tr[i]) / length
        
        # Highest high and lowest low over period
        highest_high = np.zeros_like(close_prices)
        lowest_low = np.zeros_like(close_prices)
        for i in range(length-1, len(close_prices)):
            highest_high[i] = np.max(high_prices[i-length+1:i+1])
            lowest_low[i] = np.min(low_prices[i-length+1:i+1])
        
        # Chop calculation
        chop = np.zeros_like(close_prices)
        for i in range(length-1, len(close_prices)):
            if highest_high[i] != lowest_low[i]:
                log_sum = np.sum(np.log10(atr[i-length+1:i+1] / (highest_high[i] - lowest_low[i])))
                chop[i] = 100 * log_sum / np.log10(length)
            else:
                chop[i] = 50
        return chop
    
    # Calculate indicators
    kama = calculate_kama(close, length=10, fast=2, slow=30)
    rsi = calculate_rsi(close, length=14)
    
    # Get 1d data for Choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    chop = calculate_chop(high_1d, low_1d, close_1d, length=14)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 14)  # Ensure KAMA, RSI and Chop are ready
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(kama[i]) or np.isnan(rsi[i]) or 
            np.isnan(chop_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above KAMA AND RSI < 30 (oversold) AND trending market (CHOP < 38.2)
            if (close[i] > kama[i] and 
                rsi[i] < 30 and 
                chop_aligned[i] < 38.2):
                signals[i] = 0.25
                position = 1
            # Enter short: price below KAMA AND RSI > 70 (overbought) AND trending market (CHOP < 38.2)
            elif (close[i] < kama[i] and 
                  rsi[i] > 70 and 
                  chop_aligned[i] < 38.2):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below KAMA OR RSI > 70 (overbought) OR ranging market (CHOP > 61.8)
            if (close[i] < kama[i] or 
                rsi[i] > 70 or 
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above KAMA OR RSI < 30 (oversold) OR ranging market (CHOP > 61.8)
            if (close[i] > kama[i] or 
                rsi[i] < 30 or 
                chop_aligned[i] > 61.8):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals