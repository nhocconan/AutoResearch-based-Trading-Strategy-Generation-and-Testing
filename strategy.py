#!/usr/bin/env python3
"""
Hypothesis: 12-hour Donchian(20) breakout with 1-day EMA50 trend and volume spike.
Long when price breaks above 20-period high with 1-day EMA50 rising and volume spike.
Short when price breaks below 20-period low with 1-day EMA50 falling and volume spike.
Exit when price crosses the 10-period moving average.
Donchian provides trend-following structure; 1-day EMA50 filters trend direction;
volume spike confirms institutional participation. Designed for low trade frequency by requiring
multiple confirmations and using 12h-level breakouts. Works in both bull and bear markets
by following the daily trend.
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
    
    # Load 1-day data for EMA50 trend filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 1-day EMA50 for trend filter
    close_1d = df_1d['close'].values
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period high/low) - calculated on 12h data
    high_max_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_min_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 10-period moving average for exit
    ma_10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Volume confirmation: current volume > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after enough data for EMA50
        # Skip if data not ready
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(high_max_20[i]) or 
            np.isnan(low_min_20[i]) or np.isnan(ma_10[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_spike = volume[i] > 2.0 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above 20-period high with 1-day EMA50 rising and volume spike
            if (close[i] > high_max_20[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below 20-period low with 1-day EMA50 falling and volume spike
            elif (close[i] < low_min_20[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and vol_spike):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses 10-period moving average
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below 10-period MA
                if close[i] < ma_10[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above 10-period MA
                if close[i] > ma_10[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian_20_1dEMA50_Trend_Volume"
timeframe = "12h"
leverage = 1.0