#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data for Donchian (ONCE before loop)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 5:
        return np.zeros(n)
    
    # Calculate Donchian channels on 12h (20-period)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    high_roll = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian to 12h timeframe
    upper_12h_aligned = align_htf_to_ltf(prices, df_12h, high_roll)
    lower_12h_aligned = align_htf_to_ltf(prices, df_12h, low_roll)
    
    # Volume confirmation: 5-period average
    vol_avg_5 = pd.Series(volume).rolling(window=5, min_periods=5).mean().values
    
    # Trend filter: 12h EMA21
    ema_21_12h = pd.Series(df_12h['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_21_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_21_12h)
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if data not ready or outside session
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or 
            np.isnan(ema_21_12h_aligned[i]) or np.isnan(vol_avg_5[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above upper Donchian with volume AND above EMA21 (uptrend)
            if (close[i] > upper_12h_aligned[i] and volume[i] > 1.5 * vol_avg_5[i] and 
                close[i] > ema_21_12h_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower Donchian with volume AND below EMA21 (downtrend)
            elif (close[i] < lower_12h_aligned[i] and volume[i] > 1.5 * vol_avg_5[i] and 
                  close[i] < ema_21_12h_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses back to opposite Donchian level
            if position == 1:
                if not np.isnan(lower_12h_aligned[i]) and close[i] < lower_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if not np.isnan(upper_12h_aligned[i]) and close[i] > upper_12h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "12H_Donchian20_Breakout_12hEMA21_Trend_Volume_Session"
timeframe = "12h"
leverage = 1.0