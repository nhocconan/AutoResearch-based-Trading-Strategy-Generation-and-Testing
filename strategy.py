#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Hypothesis: 12h Donchian(20) breakout with 1d ATR filter and volume confirmation
    # Donchian breakout captures trends, ATR filter avoids whipsaws in low volatility
    # Volume > 1.3x 20-period average confirms institutional participation
    # Target: 12-25 trades/year (50-100 total over 4 years) for low fee drag
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for ATR calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d ATR(14)
    tr_1d = np.zeros(len(df_1d))
    tr_1d[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(df_1d)):
        tr_1d[i] = max(high_1d[i] - low_1d[i], 
                       abs(high_1d[i] - close_1d[i-1]),
                       abs(low_1d[i] - close_1d[i-1]))
    
    atr_1d = np.zeros(len(df_1d))
    atr_1d[:14] = np.nan
    for i in range(14, len(df_1d)):
        atr_1d[i] = (atr_1d[i-1] * 13 + tr_1d[i]) / 14
    
    # Get 12h Donchian(20) for breakout
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(20, n):
        donchian_high[i] = np.max(high[i-20:i])
        donchian_low[i] = np.min(low[i-20:i])
    
    # Get 12h volume for confirmation (>1.3x 20-period average)
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.3 * vol_ma)
    
    # Align indicators to LTF (12h)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(atr_aligned[i]) or np.isnan(volume_spike[i]) or
            atr_aligned[i] <= 0):
            signals[i] = 0.0
            continue
        
        # Breakout conditions
        long_breakout = close[i] > donchian_high[i]
        short_breakout = close[i] < donchian_low[i]
        
        # ATR filter: avoid breakouts in extremely low volatility
        atr_threshold = atr_aligned[i] * 0.5  # Only trade if ATR > 0.5 * current ATR
        volatility_filter = atr_aligned[i] > atr_threshold
        
        # Entry logic: Breakout + volume confirmation + volatility filter
        long_entry = long_breakout and volume_spike[i] and volatility_filter
        short_entry = short_breakout and volume_spike[i] and volatility_filter
        
        # Exit logic: opposite breakout or volatility collapse
        long_exit = short_breakout or (atr_aligned[i] < atr_threshold * 0.5)
        short_exit = long_breakout or (atr_aligned[i] < atr_threshold * 0.5)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "12h_1d_donchian_breakout_atr_volume_filter_v1"
timeframe = "12h"
leverage = 1.0