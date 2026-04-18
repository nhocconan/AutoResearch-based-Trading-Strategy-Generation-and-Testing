#!/usr/bin/env python3
"""
Hypothesis: 12h-based strategy combining 1w Donchian(20) breakout with 1d EMA(34) trend filter,
volume confirmation, and ATR(14) stoploss. Designed to capture breakouts in both bull and bear
markets by using the 1d EMA to filter direction, targeting 15-30 trades/year to minimize fee drag.
"""
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1w data for Donchian(20) channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate Donchian(20) channels on 1w
    highest_high_1w = np.full(len(high_1w), np.nan)
    lowest_low_1w = np.full(len(low_1w), np.nan)
    for i in range(20, len(high_1w)):
        highest_high_1w[i] = np.max(high_1w[i-20:i])
        lowest_low_1w[i] = np.min(low_1w[i-20:i])
    
    # Align 1w Donchian to 12h timeframe
    highest_high_1w_12h = align_htf_to_ltf(prices, df_1w, highest_high_1w)
    lowest_low_1w_12h = align_htf_to_ltf(prices, df_1w, lowest_low_1w)
    
    # Get 1d data for EMA(34) trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close
    ema_34_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 34:
        ema_34_1d[33] = np.mean(close_1d[:34])
        for i in range(34, len(close_1d)):
            ema_34_1d[i] = (close_1d[i] * 2/35) + (ema_34_1d[i-1] * 33/35)
    
    # Align 1d EMA to 12h timeframe
    ema_34_1d_12h = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 12h ATR(14)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    atr = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            atr[i] = np.mean(tr[:14])
        else:
            atr[i] = (tr[i] * 1/14) + (atr[i-1] * 13/14)
    
    # Calculate volume moving average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20)  # need EMA, ATR, volume MA, Donchian
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_34_1d_12h[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(highest_high_1w_12h[i]) or np.isnan(lowest_low_1w_12h[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.8 * 20-period average
        vol_confirmed = volume[i] > 1.8 * vol_ma[i]
        
        # Trend filter: price above/below 1d EMA34
        trend_up = close[i] > ema_34_1d_12h[i]
        trend_down = close[i] < ema_34_1d_12h[i]
        
        if position == 0:
            # Long entry: close above 1w Donchian upper + 0.2*ATR, with volume and trend filter
            if (close[i] > highest_high_1w_12h[i] + 0.2 * atr[i] and 
                vol_confirmed and 
                trend_up):
                signals[i] = 0.25
                position = 1
            # Short entry: close below 1w Donchian lower - 0.2*ATR, with volume and trend filter
            elif (close[i] < lowest_low_1w_12h[i] - 0.2 * atr[i] and 
                  vol_confirmed and 
                  trend_down):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:
            # Long exit: close below 1w Donchian lower or ATR-based stop
            if close[i] < lowest_low_1w_12h[i] - 0.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: close above 1w Donchian upper or ATR-based stop
            if close[i] > highest_high_1w_12h[i] + 0.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Donchian20_1dEMA34_VolumeFilter"
timeframe = "12h"
leverage = 1.0