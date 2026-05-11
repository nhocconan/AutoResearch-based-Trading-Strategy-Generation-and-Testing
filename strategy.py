#!/usr/bin/env python3
name = "12h_Donchian_Breakout_DailyTrend_Volume_v2"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate ATR (20) for stoploss (not used in entry, but for potential exit)
    tr = np.maximum(high[1:] - low[1:], np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])))
    tr = np.concatenate([[high[0] - low[0]], tr])
    atr = np.zeros_like(close)
    for i in range(len(close)):
        if i < 20:
            atr[i] = np.mean(tr[:i+1])
        else:
            atr[i] = np.mean(tr[i-19:i+1])
    
    # Donchian Channel (20) for breakout
    donchian_high = np.full_like(close, np.nan)
    donchian_low = np.full_like(close, np.nan)
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA 50 for trend
    ema50_1d = np.zeros_like(df_1d['close'])
    ema50_1d[0] = df_1d['close'][0]
    alpha_50 = 2 / (50 + 1)
    for i in range(1, len(df_1d)):
        ema50_1d[i] = alpha_50 * df_1d['close'][i] + (1 - alpha_50) * ema50_1d[i-1]
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = np.zeros_like(volume)
    for i in range(len(volume)):
        if i < 20:
            vol_ma20[i] = np.mean(volume[:i+1])
        else:
            vol_ma20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Close > Donchian Upper AND Volume > 1.5x MA AND Price > Daily EMA50
            if (close[i] > donchian_high[i] and 
                volume[i] > 1.5 * vol_ma20[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Close < Donchian Lower AND Volume > 1.5x MA AND Price < Daily EMA50
            elif (close[i] < donchian_low[i] and 
                  volume[i] > 1.5 * vol_ma20[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Close < Donchian Lower (reversal) OR Volume drops below average
            if (close[i] < donchian_low[i] or volume[i] < 0.5 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25  # maintain position
        elif position == -1:
            # Short exit: Close > Donchian Upper (reversal) OR Volume drops below average
            if (close[i] > donchian_high[i] or volume[i] < 0.5 * vol_ma20[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25  # maintain position
    
    return signals