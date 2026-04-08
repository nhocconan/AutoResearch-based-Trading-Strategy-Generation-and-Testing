#!/usr/bin/env python3
# 6h_rsi_donchian_triple_screen_v1
# Hypothesis: Triple screen system on 6h timeframe combining RSI momentum, Donchian breakout, and volume confirmation.
# Long when: RSI(14) > 50 (momentum), price breaks above Donchian(20) high (breakout), volume > 1.5x 20-period average (confirmation).
# Short when: RSI(14) < 50, price breaks below Donchian(20) low, volume > 1.5x average.
# Exit when RSI crosses back below/above 50 or price retests opposite Donchian band.
# Uses 1d timeframe for trend filter: only take longs when price > 1d EMA(50), shorts when price < 1d EMA(50).
# Designed to work in both bull and bear markets via RSI momentum filter and multi-timeframe alignment.
# Target: 15-30 trades/year with strict multi-condition entry.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_rsi_donchian_triple_screen_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA trend filter (calculate once before loop)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA(50) for trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = np.full(len(close_1d), np.nan)
    for i in range(49, len(close_1d)):
        ema_50_1d[i] = np.mean(close_1d[i-49:i+1])  # Simple MA for compatibility
    
    # Align 1d EMA to 6h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # RSI(14) calculation
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    avg_gain = np.full(n, np.nan)
    avg_loss = np.full(n, np.nan)
    for i in range(14, n):
        if i == 14:
            avg_gain[i] = np.mean(gain[1:15])
            avg_loss[i] = np.mean(loss[1:15])
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.divide(avg_gain, avg_loss, out=np.full_like(avg_gain, np.nan), where=avg_loss!=0)
    rsi = np.where(avg_loss != 0, 100 - (100 / (1 + rs)), 50)
    
    # Donchian(20) channels
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    vol_surge = np.full(n, False)
    for i in range(n):
        if not np.isnan(vol_ma[i]) and vol_ma[i] > 0:
            vol_surge[i] = volume[i] > 1.5 * vol_ma[i]
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = max(20, 20)  # Donchian period
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(rsi[i]) or np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: RSI < 50 or price retests lower Donchian band
            if rsi[i] < 50 or close[i] <= donchian_low[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: RSI > 50 or price retests upper Donchian band
            if rsi[i] > 50 or close[i] >= donchian_high[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: RSI > 50, price breaks above Donchian high, volume surge, price > 1d EMA(50)
            if (rsi[i] > 50 and close[i] > donchian_high[i] and vol_surge[i] and 
                close[i] > ema_50_1d_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: RSI < 50, price breaks below Donchian low, volume surge, price < 1d EMA(50)
            elif (rsi[i] < 50 and close[i] < donchian_low[i] and vol_surge[i] and 
                  close[i] < ema_50_1d_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals