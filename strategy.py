#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 20-period Donchian channels on 1d
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Align to 4h timeframe
    highest_20_4h = align_htf_to_ltf(prices, df_1d, highest_20)
    lowest_20_4h = align_htf_to_ltf(prices, df_1d, lowest_20)
    
    # Calculate 4h RSI(14) for momentum filter
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = pd.Series(gain).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    avg_loss = pd.Series(loss).ewm(alpha=1/14, adjust=False, min_periods=14).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values = rsi.values
    
    # Volume filter: current volume > 1.5 * 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 20  # Need Donchian and RSI
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(highest_20_4h[i]) or 
            np.isnan(lowest_20_4h[i]) or 
            np.isnan(rsi_values[i]) or 
            np.isnan(volume_ma20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > (1.5 * volume_ma20[i])
        
        # Momentum filter: RSI not in extreme overbought/oversold
        rsi_not_extreme = (rsi_values[i] > 30) and (rsi_values[i] < 70)
        
        if position == 0:
            # Long: price breaks above 1d Donchian high with volume and momentum
            if (close[i] > highest_20_4h[i]) and volume_filter and rsi_not_extreme:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below 1d Donchian low with volume and momentum
            elif (close[i] < lowest_20_4h[i]) and volume_filter and rsi_not_extreme:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 1d Donchian low
            if close[i] < lowest_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 1d Donchian high
            if close[i] > highest_20_4h[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_1D_Breakout_Volume_Momentum"
timeframe = "4h"
leverage = 1.0