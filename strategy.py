#!/usr/bin/env python3
"""
6h_1w_1d_Stochastic_RSI_Divergence_ML
Hypothesis: Combine weekly Stochastic RSI extremes with daily momentum divergence on 6h timeframe to capture trend exhaustion points. Weekly Stochastic RSI >80 or <20 identifies overbought/oversold conditions on higher timeframe, while daily RSI divergence (price making new high/low but RSI not confirming) signals weakening momentum. This works in both bull and bear markets by fading extremes when higher timeframe momentum is exhausted, reducing whipsaws. Targets 15-25 trades/year by requiring weekly extreme + daily divergence + volume confirmation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_rsi(prices, period=14):
    """Calculate RSI with proper handling"""
    delta = np.diff(prices)
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.zeros_like(prices)
    avg_loss = np.zeros_like(prices)
    
    # Wilder's smoothing
    avg_gain[period] = np.mean(gain[:period])
    avg_loss[period] = np.mean(loss[:period])
    
    for i in range(period + 1, len(prices)):
        avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:period] = np.nan
    return rsi

def calculate_stoch_rsi(rsi, period=14):
    """Calculate Stochastic RSI"""
    k = np.full_like(rsi, np.nan)
    d = np.full_like(rsi, np.nan)
    
    for i in range(period, len(rsi)):
        if not np.isnan(rsi[i-period:i+1]).any():
            min_rsi = np.nanmin(rsi[i-period:i+1])
            max_rsi = np.nanmax(rsi[i-period:i+1])
            if max_rsi != min_rsi:
                k[i] = (rsi[i] - min_rsi) / (max_rsi - min_rsi) * 100
    
    # Smooth %K to get %D (3-period SMA of %K)
    for i in range(2, len(k)):
        if not np.isnan(k[i-2:i+1]).any():
            d[i] = np.nanmean(k[i-2:i+1])
    
    return k, d

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Stochastic RSI (HTF)
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly RSI then Stochastic RSI
    rsi_1w = calculate_rsi(close_1w, 14)
    stoch_k_1w, stoch_d_1w = calculate_stoch_rsi(rsi_1w, 14)
    
    # Align weekly Stochastic RSI to 6h timeframe
    stoch_k_1w_aligned = align_htf_to_ltf(prices, df_1w, stoch_k_1w)
    stoch_d_1w_aligned = align_htf_to_ltf(prices, df_1w, stoch_d_1w)
    
    # Get daily data for RSI divergence (HTF)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate daily RSI
    rsi_1d = calculate_rsi(close_1d, 14)
    
    # Align daily RSI to 6h timeframe
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Volume confirmation: current volume > 1.3 x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    vol_confirm = volume > (vol_ma * 1.3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(30, 20)  # need indicators warmed up
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(stoch_k_1w_aligned[i]) or np.isnan(stoch_d_1w_aligned[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Detect daily RSI divergence
        bullish_div = False
        bearish_div = False
        
        # Look back 5 periods for divergence
        lookback = 5
        if i >= lookback:
            # Bullish divergence: price makes lower low, RSI makes higher low
            if (low[i] < low[i-lookback] and 
                rsi_1d_aligned[i] > rsi_1d_aligned[i-lookback]):
                bullish_div = True
            # Bearish divergence: price makes higher high, RSI makes lower high
            elif (high[i] > high[i-lookback] and 
                  rsi_1d_aligned[i] < rsi_1d_aligned[i-lookback]):
                bearish_div = True
        
        if position == 0:
            # Long entry: weekly Stochastic RSI oversold (<20) + bullish divergence + volume
            if (stoch_k_1w_aligned[i] < 20 and stoch_d_1w_aligned[i] < 20 and
                bullish_div and vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short entry: weekly Stochastic RSI overbought (>80) + bearish divergence + volume
            elif (stoch_k_1w_aligned[i] > 80 and stoch_d_1w_aligned[i] > 80 and
                  bearish_div and vol_confirm[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: weekly Stochastic RSI overbought (>80) or bearish divergence
            if (stoch_k_1w_aligned[i] > 80 or bearish_div):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: weekly Stochastic RSI oversold (<20) or bullish divergence
            if (stoch_k_1w_aligned[i] < 20 or bullish_div):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_1w_1d_Stochastic_RSI_Divergence_ML"
timeframe = "6h"
leverage = 1.0