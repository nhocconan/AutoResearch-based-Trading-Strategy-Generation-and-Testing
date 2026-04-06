#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "exp_14064_1d_donchian20_1w_ema_vol_v1"
timeframe = "1d"
leverage = 1.0

def calculate_ema(arr, period):
    return pd.Series(arr).ewm(span=period, adjust=False, min_periods=period).mean().values

def calculate_atr(high, low, close, period):
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(np.maximum(tr1, tr2), tr3)
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False, min_periods=period).mean().values
    return atr

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load weekly data for EMA filter (once before loop)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 50 EMA on weekly
    ema_50_1w = calculate_ema(close_1w, 50)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Daily data for Donchian channels and ATR
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 20-period Donchian channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Calculate 20-period average volume
    avg_vol_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # ATR for stop loss (14-period)
    atr = calculate_atr(high, low, close, 14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    stop_price = 0.0
    
    # Start from warmup period (max of 50 for weekly EMA, 20 for Donchian, 14 for ATR)
    start = max(50, 20, 14)
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(avg_vol_20[i]) or \
           np.isnan(ema_50_1w_aligned[i]) or np.isnan(atr[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check stops
        if position == 1:  # long position
            # Check stop loss
            if close[i] <= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        elif position == -1:  # short position
            # Check stop loss
            if close[i] >= stop_price:
                signals[i] = 0.0
                position = 0
                continue
        
        # Donchian breakout signals with volume confirmation and weekly EMA filter
        volume_surge = volume[i] > 1.5 * avg_vol_20[i]  # 50% volume surge
        breakout_up = high[i] > high_20[i-1]  # Break above 20-period high
        breakout_down = low[i] < low_20[i-1]  # Break below 20-period low
        
        # Weekly EMA filter: only long above weekly EMA, short below
        price_above_weekly_ema = close[i] > ema_50_1w_aligned[i]
        price_below_weekly_ema = close[i] < ema_50_1w_aligned[i]
        
        # Generate signals
        if position == 0:
            # Long: breakout up + volume surge + price above weekly EMA
            if breakout_up and volume_surge and price_above_weekly_ema:
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
                stop_price = entry_price - (2.0 * atr[i])
            # Short: breakout down + volume surge + price below weekly EMA
            elif breakout_down and volume_surge and price_below_weekly_ema:
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
                stop_price = entry_price + (2.0 * atr[i])
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long on stop or breakdown
            if close[i] <= stop_price or low[i] < low_20[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short on stop or breakout
            if close[i] >= stop_price or high[i] > high_20[i-1]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals