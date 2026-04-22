#!/usr/bin/env python3
"""
Hypothesis: 1-day Donchian breakout with 1-week trend filter and volume confirmation.
Long when price breaks above Donchian(20) high, weekly EMA(21) is rising, and daily volume > 20-day average.
Short when price breaks below Donchian(20) low, weekly EMA(21) is falling, and daily volume > 20-day average.
Exit when price crosses back below Donchian(20) midpoint (for longs) or above midpoint (for shorts).
Uses weekly trend to filter direction and volume to confirm institutional participation.
Works in bull markets via breakouts and in bear via short breakdowns with trend alignment.
Target: 20-40 trades/year to minimize fee drag.
"""

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
    
    # Load daily data for Donchian and volume filter - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Donchian channels (20-period)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2.0
    
    # Daily volume average (20-period)
    vol_1d = df_1d['volume'].values
    avg_vol_20 = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    
    # Load weekly data for EMA trend filter - ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_21_1w = pd.Series(close_1w).ewm(span=21, min_periods=21, adjust=False).mean().values
    
    # Align all HTF data to lower timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_1d, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_1d, donchian_low)
    donchian_mid_aligned = align_htf_to_ltf(prices, df_1d, donchian_mid)
    avg_vol_20_aligned = align_htf_to_ltf(prices, df_1d, avg_vol_20)
    ema_21_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_21_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):
        # Skip if data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(donchian_mid_aligned[i]) or np.isnan(avg_vol_20_aligned[i]) or
            np.isnan(ema_21_1w_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Price breaks above Donchian high, weekly EMA rising, volume above average
            if (close[i] > donchian_high_aligned[i] and 
                ema_21_1w_aligned[i] > ema_21_1w_aligned[i-1] and
                volume[i] > avg_vol_20_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low, weekly EMA falling, volume above average
            elif (close[i] < donchian_low_aligned[i] and 
                  ema_21_1w_aligned[i] < ema_21_1w_aligned[i-1] and
                  volume[i] > avg_vol_20_aligned[i]):
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            
            if position == 1:
                # Exit long: Price crosses below Donchian midpoint
                if close[i] < donchian_mid_aligned[i]:
                    exit_signal = True
            else:  # position == -1
                # Exit short: Price crosses above Donchian midpoint
                if close[i] > donchian_mid_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "1D_Donchian_20_1wEMA21_Volume_Filter"
timeframe = "1d"
leverage = 1.0