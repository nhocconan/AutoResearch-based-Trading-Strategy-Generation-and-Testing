#!/usr/bin/env python3
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
    
    # Get daily data for Bollinger Bands and EMA
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Bollinger Bands (20, 2)
    sma_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(close_1d)):
        sma_20_1d[i] = np.mean(close_1d[i-20:i])
    
    std_20_1d = np.full(len(df_1d), np.nan)
    for i in range(20, len(close_1d)):
        std_20_1d[i] = np.std(close_1d[i-20:i])
    
    upper_bb_1d = sma_20_1d + 2 * std_20_1d
    lower_bb_1d = sma_20_1d - 2 * std_20_1d
    
    # Weekly EMA (50)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_50_1w = np.full(len(df_1w), np.nan)
    alpha_w = 2 / (50 + 1)
    for i in range(len(close_1w)):
        if i < 49:
            ema_50_1w[i] = np.mean(close_1w[:i+1]) if i > 0 else close_1w[i]
        else:
            if np.isnan(ema_50_1w[i-1]):
                ema_50_1w[i] = np.mean(close_1w[i-49:i+1])
            else:
                ema_50_1w[i] = close_1w[i] * alpha_w + ema_50_1w[i-1] * (1 - alpha_w)
    
    # Align to 6h timeframe
    sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, sma_20_1d)
    upper_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, upper_bb_1d)
    lower_bb_1d_aligned = align_htf_to_ltf(prices, df_1d, lower_bb_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    position = 0
    
    # Warmup
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if (np.isnan(sma_20_1d_aligned[i]) or 
            np.isnan(upper_bb_1d_aligned[i]) or
            np.isnan(lower_bb_1d_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        
        if position == 0:
            # Long: Price breaks above upper BB + price above weekly EMA
            if (price > upper_bb_1d_aligned[i] and 
                price > ema_50_1w_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower BB + price below weekly EMA
            elif (price < lower_bb_1d_aligned[i] and 
                  price < ema_50_1w_aligned[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long exit: Price crosses below SMA or weekly EMA turns down
            if (price < sma_20_1d_aligned[i] or 
                ema_50_1w_aligned[i] < ema_50_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price crosses above SMA or weekly EMA turns up
            if (price > sma_20_1d_aligned[i] or 
                ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_BollingerBreakout_WeeklyEMA50_v1"
timeframe = "6h"
leverage = 1.0