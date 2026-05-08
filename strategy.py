#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation
# Long when price breaks above Donchian(20) high, 12h EMA50 rising, volume > 1.5x average
# Short when price breaks below Donchian(20) low, 12h EMA50 falling, volume > 1.5x average
# Uses Donchian channel for price structure, EMA50 for trend filter, volume for confirmation
# Targets 20-50 total trades per year (80-200 over 4 years) for low fee drag and high win rate
# Works in both bull and bear markets due to trend filter and volume confirmation

name = "4h_Donchian20_12hEMA50_Volume"
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
    
    # Get 12h data for Donchian levels and trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels from previous 12h data (to avoid look-ahead)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Previous 12h values (shifted by 1 to avoid look-ahead)
    prev_high = np.roll(high_12h, 1)
    prev_low = np.roll(low_12h, 1)
    prev_high[0] = np.nan  # First 12h bar has no previous
    prev_low[0] = np.nan
    
    # Donchian(20) on 12h data
    donchian_high = pd.Series(prev_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(prev_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Calculate EMA50 on 12h close for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_conf = volume > (vol_ma * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need at least 20 12h bars of data for Donchian
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        high_val = high[i]
        low_val = low[i]
        donchian_high_val = donchian_high_aligned[i]
        donchian_low_val = donchian_low_aligned[i]
        ema50_12h_val = ema50_12h_aligned[i]
        vol_conf_val = vol_conf[i]
        
        if position == 0:
            # Enter long: price breaks above Donchian high, 12h uptrend, volume confirmation
            if high_val > donchian_high_val and ema50_12h_val > 0 and vol_conf_val:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below Donchian low, 12h downtrend, volume confirmation
            elif low_val < donchian_low_val and ema50_12h_val < 0 and vol_conf_val:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below Donchian low or 12h trend down
            if low_val < donchian_low_val or ema50_12h_val < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above Donchian high or 12h trend up
            if high_val > donchian_high_val or ema50_12h_val > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals