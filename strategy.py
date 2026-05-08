#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d RSI filter and volume spike
# Donchian breakouts capture momentum in trending markets, while 1d RSI avoids extreme overbought/oversold conditions.
# Volume spike confirms breakout strength. Designed for 15-35 trades/year (~60-140 total over 4 years) to minimize fee drag.
# Works in both bull and breakouts in bear markets (e.g., 2023 rallies).

name = "4h_Donchian20_1dRSI_Volume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Donchian channels: 20-period high/low
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1d RSI for trend filter (avoid extremes)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    delta = pd.Series(close_1d).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi_1d = 100 - (100 / (1 + rs))
    rsi_1d = rsi_1d.fillna(50).values  # neutral when undefined
    rsi_1d = np.where(np.isnan(rsi_1d), 50, rsi_1d)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need enough data for Donchian and RSI
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(rsi_1d[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Get aligned 1d RSI value
        rsi_1d_val = rsi_1d[i]  # rsi_1d is already computed on 1d data, need to align
        # Actually compute aligned array properly
        if i == start_idx:  # Compute aligned arrays once outside loop would be better, but for clarity:
            # We'll compute the aligned rsi array properly below
            pass
    
    # Properly compute aligned 1d RSI
    rsi_1d_aligned = align_htf_to_ltf(prices, df_1d, rsi_1d)
    
    signals = np.zeros(n)
    position = 0
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or 
            np.isnan(rsi_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above Donchian upper, RSI not overbought (<70), volume confirmation
            if close[i] > high_20[i] and rsi_1d_aligned[i] < 70 and vol_conf[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian lower, RSI not oversold (>30), volume confirmation
            elif close[i] < low_20[i] and rsi_1d_aligned[i] > 30 and vol_conf[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian lower or RSI becomes overbought
            if close[i] < low_20[i] or rsi_1d_aligned[i] > 70:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian upper or RSI becomes oversold
            if close[i] > high_20[i] or rsi_1d_aligned[i] < 30:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals