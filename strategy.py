#!/usr/bin/env python3
"""
1d Weekly Donchian Breakout with Volume Spike and RSI Filter
Breakout above/below weekly Donchian channels + volume spike + daily RSI filter
Designed to capture long-term trends with low trade frequency
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    high_20 = pd.Series(high_1w).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_1w).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian to daily
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Get daily data for RSI
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate daily RSI (14-period)
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Align daily RSI to daily (no shift needed as it's already daily)
    rsi_aligned = rsi_values  # Already at daily frequency
    
    # Volume spike detection (2x 20-day average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # -1 short, 0 flat, 1 long
    
    start_idx = 40  # need enough history for calculations
    
    for i in range(start_idx, n):
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        upper_channel = high_20_aligned[i]
        lower_channel = low_20_aligned[i]
        rsi_val = rsi_aligned[i]
        
        if position == 0:
            # Long: breakout above weekly upper channel + volume spike + RSI > 50
            if (price > upper_channel and 
                volume_spike[i] and 
                rsi_val > 50):
                signals[i] = 0.25
                position = 1
            # Short: breakout below weekly lower channel + volume spike + RSI < 50
            elif (price < lower_channel and 
                  volume_spike[i] and 
                  rsi_val < 50):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below weekly lower channel or RSI < 40
            if price < lower_channel or rsi_val < 40:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above weekly upper channel or RSI > 60
            if price > upper_channel or rsi_val > 60:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "1d_WeeklyDonchian_Breakout_Volume_RSI"
timeframe = "1d"
leverage = 1.0