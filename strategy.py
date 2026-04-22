#!/usr/bin/env python3
"""
Hypothesis: 4-hour Donchian breakout (20-period) with 1-day EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian upper band with 1-day EMA50 rising and volume > 1.5x average.
Short when price breaks below Donchian lower band with 1-day EMA50 falling and volume > 1.5x average.
Exit when price crosses the 10-period moving average in the opposite direction.
Designed for low trade frequency by requiring multiple confirmations and using daily trend filter.
Works in both bull and bear markets by following the daily trend direction.
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
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Donchian channels (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Exit condition: 10-period moving average
    ma10 = pd.Series(close).rolling(window=10, min_periods=10).mean().values
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after enough data for Donchian
        # Skip if data not ready
        if (np.isnan(donchian_upper[i]) or np.isnan(donchian_lower[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(ma10[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation
        vol_confirmed = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: Price breaks above Donchian upper with 1-day EMA50 rising and volume confirmation
            if (close[i] > donchian_upper[i] and 
                ema50_1d_aligned[i] > ema50_1d_aligned[i-1] and vol_confirmed):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian lower with 1-day EMA50 falling and volume confirmation
            elif (close[i] < donchian_lower[i] and 
                  ema50_1d_aligned[i] < ema50_1d_aligned[i-1] and vol_confirmed):
                signals[i] = -0.25
                position = -1
        else:
            # Exit: Price crosses 10-period MA in opposite direction
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below MA10
                if close[i] < ma10[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above MA10
                if close[i] > ma10[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4H_Donchian_20_1dEMA50_Trend_Volume"
timeframe = "4h"
leverage = 1.0