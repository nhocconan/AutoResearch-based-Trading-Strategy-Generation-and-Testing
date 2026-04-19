#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA200 trend filter.
# Long when price breaks above 20-bar high, volume > 1.5x average, and above daily EMA200.
# Short when price breaks below 20-bar low, volume > 1.5x average, and below daily EMA200.
# Exit on opposite Donchian break (10-bar) or trend reversal.
# Designed to work in both bull and bear markets by filtering with trend and volume.
# Target: 20-40 trades/year per symbol.
name = "4h_Donchian20_Volume_EMA200_Trend"
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
    
    # Get 1d data for EMA200 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Donchian channels (20-period)
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Donchian channels (10-period) for exit
    high_max_10 = pd.Series(high).rolling(window=10, min_periods=10).max().values
    low_min_10 = pd.Series(low).rolling(window=10, min_periods=10).min().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Align 1d EMA200 to 4h
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(200, 20)  # Ensure EMA200 and Donchian are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        ema_200_val = ema_200_aligned[i]
        vol_ma = vol_ma_20[i]
        vol = volume[i]
        
        # Volume confirmation threshold
        volume_confirmed = vol > 1.5 * vol_ma
        
        if position == 0:
            # Enter long if price breaks above 20-bar high, volume confirmation, and above daily EMA200
            if price > high_max_20[i] and volume_confirmed and price > ema_200_val:
                signals[i] = 0.25
                position = 1
            # Enter short if price breaks below 20-bar low, volume confirmation, and below daily EMA200
            elif price < low_min_20[i] and volume_confirmed and price < ema_200_val:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long when price breaks below 10-bar low or crosses below daily EMA200
            if price < low_min_10[i] or price < ema_200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short when price breaks above 10-bar high or crosses above daily EMA200
            if price > high_max_10[i] or price > ema_200_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals