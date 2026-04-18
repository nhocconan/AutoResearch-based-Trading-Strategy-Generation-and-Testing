#!/usr/bin/env python3
"""
1d_KAMA_Direction_RSI_ChopFilter
1d strategy using KAMA direction as primary trend filter, RSI for overbought/oversold,
and Choppiness Index to avoid ranging markets. Enters long when KAMA trending up,
RSI < 40, and Chop > 61.8 (ranging). Enters short when KAMA trending down,
RSI > 60, and Chop > 61.8. Exits on opposite signal.
Designed for ~10-20 trades/year per symbol (40-80 total over 4 years).
Works in bull markets (KAMA up + RSI pullback) and bear markets (KAMA down + RSI bounce).
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
    
    # Get weekly data for Choppiness Index (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Choppiness Index on weekly data
    def choppiness_index(high, low, close, period=14):
        atr = np.zeros_like(close)
        for i in range(1, len(close)):
            atr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )
        # Smoothed ATR (using simple moving average for simplicity)
        atr_ma = np.zeros_like(close)
        for i in range(period, len(close)):
            atr_ma[i] = np.mean(atr[i-period+1:i+1])
        
        # Highest high and lowest low over period
        highest_high = np.zeros_like(close)
        lowest_low = np.zeros_like(close)
        for i in range(period-1, len(close)):
            highest_high[i] = np.max(high[i-period+1:i+1])
            lowest_low[i] = np.min(low[i-period+1:i+1])
        
        # Chop calculation
        chop = np.full_like(close, 50.0)
        for i in range(period-1, len(close)):
            if atr_ma[i] > 0 and (highest_high[i] - lowest_low[i]) > 0:
                chop[i] = 100 * np.log10(np.sum(atr[i-period+1:i+1]) / (highest_high[i] - lowest_low[i])) / np.log10(period)
        return chop
    
    chop_1w = choppiness_index(high_1w, low_1w, close_1w, 14)
    chop_1w_aligned = align_htf_to_ltf(prices, df_1w, chop_1w)
    
    # Get daily data for KAMA and RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # KAMA calculation
    def kama(close, period=10, fast=2, slow=30):
        # Efficiency Ratio
        change = np.abs(np.diff(close, n=period))
        volatility = np.sum(np.abs(np.diff(close)), axis=0) if len(close) > 1 else 0
        er = np.zeros_like(close)
        for i in range(period, len(close)):
            if volatility[i] > 0:
                er[i] = change[i] / volatility[i]
            else:
                er[i] = 0
        
        # Smoothing constants
        sc = (er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)) ** 2
        
        # KAMA
        kama_vals = np.zeros_like(close)
        kama_vals[0] = close[0]
        for i in range(1, len(close)):
            kama_vals[i] = kama_vals[i-1] + sc[i] * (close[i] - kama_vals[i-1])
        return kama_vals
    
    kama_1d = kama(close_1d, 10, 2, 30)
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    
    # RSI calculation
    def rsi(close, period=14):
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.zeros_like(close)
        avg_loss = np.zeros_like(close)
        
        # First average
        avg_gain[period] = np.mean(gain[:period])
        avg_loss[period] = np.mean(loss[:period])
        
        # Subsequent averages
        for i in range(period+1, len(close)):
            avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i-1]) / period
            avg_loss[i] = (avg_loss[i-1] * (period-1) + loss[i-1]) / period
        
        rs = np.zeros_like(close)
        rsi_vals = np.zeros_like(close)
        for i in range(period, len(close)):
            if avg_loss[i] != 0:
                rs[i] = avg_gain[i] / avg_loss[i]
                rsi_vals[i] = 100 - (100 / (1 + rs[i]))
            else:
                rsi_vals[i] = 100
        return rsi_vals
    
    rsi_1d = rsi(close_1d, 14)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(chop_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        # KAMA direction (using 2-period slope)
        kama_up = kama_1d_aligned[i] > kama_1d_aligned[i-1]
        kama_down = kama_1d_aligned[i] < kama_1d_aligned[i-1]
        
        # Chop filter: only trade in ranging markets (Chop > 61.8)
        chop_filter = chop_1w_aligned[i] > 61.8
        
        if position == 0:
            # Long: KAMA up, RSI oversold (<40), ranging market
            if kama_up and rsi_1d_aligned[i] < 40 and chop_filter:
                signals[i] = 0.25
                position = 1
            # Short: KAMA down, RSI overbought (>60), ranging market
            elif kama_down and rsi_1d_aligned[i] > 60 and chop_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA down or RSI overbought
            if kama_down or rsi_1d_aligned[i] > 60:
                signals[i] = -0.25  # reverse to short
                position = -1
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA up or RSI oversold
            if kama_up or rsi_1d_aligned[i] < 40:
                signals[i] = 0.25  # reverse to long
                position = 1
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_KAMA_Direction_RSI_ChopFilter"
timeframe = "1d"
leverage = 1.0