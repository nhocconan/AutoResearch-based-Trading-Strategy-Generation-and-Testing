#!/usr/bin/env python3
"""
Hypothesis: Daily Donchian breakout with 1-week EMA filter and volume confirmation.
Long when price breaks above Donchian(20) high, closes above 1-week EMA50, and volume > 1.5x average.
Short when price breaks below Donchian(20) low, closes below 1-week EMA50, and volume > 1.5x average.
Exit when price crosses opposite Donchian boundary or closes below/above 1-week EMA50.
Uses daily timeframe for low trade frequency and 1-week EMA for trend filter.
Designed to capture strong trends while avoiding false breakouts in ranging markets.
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
    
    # Load 1-week data for EMA50 trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period)
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):
        # Skip if data not ready
        if (np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_avg[i]) or
            np.isnan(ema50_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, close above 1-week EMA50, volume spike
            if (high[i] > high_20[i] and 
                close[i] > ema50_1w_aligned[i] and 
                volume[i] > 1.5 * vol_avg[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, close below 1-week EMA50, volume spike
            elif (low[i] < low_20[i] and 
                  close[i] < ema50_1w_aligned[i] and 
                  volume[i] > 1.5 * vol_avg[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price breaks below Donchian low OR close below 1-week EMA50
                if (low[i] < low_20[i] or 
                    close[i] < ema50_1w_aligned[i]):
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price breaks above Donchian high OR close above 1-week EMA50
                if (high[i] > high_20[i] or 
                    close[i] > ema50_1w_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian_20_1wEMA50_Volume"
timeframe = "1d"
leverage = 1.0