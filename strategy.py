#!/usr/bin/env python3
"""
4h Donchian Breakout with Volume Spike and 1d RSI Filter
Designed for low trade frequency with strong edge in both bull and bear markets.
Uses Donchian channel breakout for entry, volume spike for confirmation, and
1d RSI > 50 for long / < 50 for short to align with higher timeframe momentum.
Focuses on clean breakouts in direction of daily trend to reduce whipsaw.
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
    
    # Get 1d data for RSI filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d RSI(14)
    delta = pd.Series(close_1d).diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values  # neutral when undefined
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    # Donchian channel (20-period) on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike (2x 4-period average)
    vol_ma = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    entry_price = 0.0
    
    start_idx = 30  # need enough history for Donchian
    
    for i in range(start_idx, n):
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper = high_20[i]
        lower = low_20[i]
        rsi_val = rsi_1d_aligned[i]
        
        if position == 0:
            # Long: break above upper band with volume spike and daily RSI > 50
            if (price > upper and 
                volume_spike[i] and 
                rsi_val > 50):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: break below lower band with volume spike and daily RSI < 50
            elif (price < lower and 
                  volume_spike[i] and 
                  rsi_val < 50):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Long position: hold until reverse signal
            signals[i] = 0.25
            if price < lower:  # reverse signal: break below lower band
                signals[i] = 0.0
                position = 0
        
        elif position == -1:
            # Short position: hold until reverse signal
            signals[i] = -0.25
            if price > upper:  # reverse signal: break above upper band
                signals[i] = 0.0
                position = 0
    
    return signals

name = "4h_Donchian_Breakout_Volume_Spike_1dRSI"
timeframe = "4h"
leverage = 1.0