#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Donchian channels (20-period) for breakout
    high20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # 12h EMA34 for trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    
    # Warmup period
    warmup = 60
    
    # Track position state
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(high20[i]) or 
            np.isnan(low20[i]) or 
            np.isnan(vol_ma[i]) or 
            np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            position = 0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_confirmed = volume[i] > 1.5 * vol_ma[i]
        
        # Entry logic: only enter when flat
        if position == 0:
            # Long: price breaks above 20-period high, above 12h EMA34, volume confirmed
            if (close[i] > high20[i] and 
                close[i] > ema_34_12h_aligned[i] and 
                vol_confirmed):
                signals[i] = 0.25
                position = 1
                continue
            # Short: price breaks below 20-period low, below 12h EMA34, volume confirmed
            elif (close[i] < low20[i] and 
                  close[i] < ema_34_12h_aligned[i] and 
                  vol_confirmed):
                signals[i] = -0.25
                position = -1
                continue
        
        # Exit logic: reverse signal or trend change
        elif position == 1:
            # Exit long: price breaks below 20-period low OR price crosses below 12h EMA34
            if (close[i] < low20[i] or 
                close[i] < ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above 20-period high OR price crosses above 12h EMA34
            if (close[i] > high20[i] or 
                close[i] > ema_34_12h_aligned[i]):
                signals[i] = 0.0
                position = 0
                continue
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Donchian_20_12hEMA34_Volume_Confirmation_v1"
timeframe = "6h"
leverage = 1.0