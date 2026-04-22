#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume confirmation
# Long when price breaks above Donchian high + price > 1d EMA50 + volume > 1.5x avg
# Short when price breaks below Donchian low + price < 1d EMA50 + volume > 1.5x avg
# Exit when price returns to Donchian mean or trend reverses
# Designed for low trade frequency (~20-40/year) with strong edge in both bull and bear markets
# Uses Donchian channels for breakout structure and EMA50 for trend filter

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Donchian channels (20-period) on 4h data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Donchian high and low (20-period)
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donch_mean = (donch_high + donch_low) / 2
    
    # Calculate 20-period average volume for volume confirmation
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if data not ready
        if (np.isnan(donch_high[i]) or 
            np.isnan(donch_low[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_ma = vol_ma_20[i]
        dh = donch_high[i]
        dl = donch_low[i]
        dm = donch_mean[i]
        ema_val = ema_50_aligned[i]
        
        # Volume filter: current volume > 1.5 * 20-day average
        vol_confirm = vol > 1.5 * vol_ma
        
        if position == 0:
            # Long conditions: price breaks above Donchian high + uptrend + volume confirmation
            if price > dh and price > ema_val and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below Donchian low + downtrend + volume confirmation
            elif price < dl and price < ema_val and vol_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit conditions: price returns to Donchian mean or trend reverses
            exit_signal = False
            
            if position == 1:  # long position
                # Exit when price returns to Donchian mean or trend turns down
                if price <= dm or price < ema_val:
                    exit_signal = True
            
            elif position == -1:  # short position
                # Exit when price returns to Donchian mean or trend turns up
                if price >= dm or price > ema_val:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Donchian20_1dEMA50_Volume_Confirmation"
timeframe = "4h"
leverage = 1.0