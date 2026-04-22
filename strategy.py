#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian(20) breakout with 12-hour EMA50 trend and volume spike.
Long when price breaks above upper band with 12-hour EMA50 rising and volume spike.
Short when price breaks below lower band with 12-hour EMA50 falling and volume spike.
Exit when price returns to middle band (SMA20).
Donchian channels provide trend-following structure; 12-hour EMA50 filters trend direction;
volume spike confirms institutional participation. Designed for low trade frequency by requiring
multiple confirmations. Works in both bull and bear markets by following the 12-hour trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12-hour data for EMA50 trend filter - ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # 12-hour EMA50 for trend filter
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Donchian(20) on 4-hour data
    upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    middle_20 = pd.Series(close).rolling(window=20, min_periods=20).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for EMA50
        # Skip if data not ready
        if (np.isnan(ema50_12h_aligned[i]) or np.isnan(upper_20[i]) or np.isnan(lower_20[i]) or 
            np.isnan(middle_20[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above upper band with 12h EMA50 rising and volume spike
            if (close[i] > upper_20[i] and 
                ema50_12h_aligned[i] > ema50_12h_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below lower band with 12h EMA50 falling and volume spike
            elif (close[i] < lower_20[i] and 
                  ema50_12h_aligned[i] < ema50_12h_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price returns to middle band
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below middle band
                if close[i] < middle_20[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above middle band
                if close[i] > middle_20[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_20_12hEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0