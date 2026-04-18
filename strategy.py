#!/usr/bin/env python3
"""
12h_KAMA_Direction_RSI_Filter_Volume_Trend
Hypothesis: On 12h timeframe, use KAMA trend direction filtered by RSI (50) and volume > 1.5x 20-period average.
Only take longs when KAMA upward and RSI > 50, shorts when KAMA downward and RSI < 50.
Exit when KAMA reverses or volume dries up.
Designed for low frequency (15-30 trades/year) with trend-following edge in both bull and bear markets.
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
    
    # Get daily data for KAMA and RSI calculation
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate daily KAMA ( Kaufman Adaptive Moving Average )
    close_1d = df_1d['close'].values
    kama_1d = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 2:
        # Efficiency ratio and smoothing constants
        fast_sc = 2 / (2 + 1)   # EMA(2)
        slow_sc = 2 / (30 + 1)  # EMA(30)
        
        kama_1d[0] = close_1d[0]
        
        for i in range(1, len(close_1d)):
            # Calculate efficiency ratio
            if i >= 10:
                change = abs(close_1d[i] - close_1d[i-10])
                volatility = 0
                for j in range(1, 11):
                    volatility += abs(close_1d[i-j+1] - close_1d[i-j])
                if volatility > 0:
                    er = change / volatility
                else:
                    er = 0
            else:
                er = 0
            
            # Smoothing constant
            sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
            
            # KAMA calculation
            kama_1d[i] = kama_1d[i-1] + sc * (close_1d[i] - kama_1d[i-1])
    
    # Calculate daily RSI(14)
    rsi_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 15:
        delta = np.diff(close_1d)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = np.full(len(close_1d), np.nan)
        avg_loss = np.full(len(close_1d), np.nan)
        
        # First average
        avg_gain[14] = np.mean(gain[1:15])
        avg_loss[14] = np.mean(loss[1:15])
        
        # Wilder smoothing
        for i in range(15, len(close_1d)):
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i-1]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i-1]) / 14
            
            rs = avg_gain[i] / avg_loss[i] if avg_loss[i] != 0 else 0
            rsi_1d[i] = 100 - (100 / (1 + rs))
    
    # Get weekly trend filter (EMA34)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    ema34_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= 34:
        ema34_1w[33] = np.mean(close_1w[0:34])
        alpha = 2 / (34 + 1)
        for i in range(34, len(close_1w)):
            ema34_1w[i] = close_1w[i] * alpha + ema34_1w[i-1] * (1 - alpha)
    
    # Align all indicators to 12h timeframe
    kama_1d_aligned = align_htf_to_ltf(prices, df_1d, kama_1d)
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    ema34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema34_1w)
    
    # Volume filter: current volume > 1.5 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_filter = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # Ensure we have enough data
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(kama_1d_aligned[i]) or np.isnan(rsi_1d_aligned[i]) or 
            np.isnan(ema34_1w_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: KAMA trending up, RSI > 50, volume filter, and above weekly EMA
            if (close[i] > kama_1d_aligned[i] and 
                rsi_1d_aligned[i] > 50 and 
                vol_filter[i] and 
                close[i] > ema34_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: KAMA trending down, RSI < 50, volume filter, and below weekly EMA
            elif (close[i] < kama_1d_aligned[i] and 
                  rsi_1d_aligned[i] < 50 and 
                  vol_filter[i] and 
                  close[i] < ema34_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: KAMA turns down or RSI < 50 or volume filter fails
            if (close[i] < kama_1d_aligned[i] or 
                rsi_1d_aligned[i] < 50 or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: KAMA turns up or RSI > 50 or volume filter fails
            if (close[i] > kama_1d_aligned[i] or 
                rsi_1d_aligned[i] > 50 or 
                not vol_filter[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_KAMA_Direction_RSI_Filter_Volume_Trend"
timeframe = "12h"
leverage = 1.0