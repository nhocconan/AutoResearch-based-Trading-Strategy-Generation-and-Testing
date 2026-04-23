#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above Donchian(20) high AND price > 1d EMA50 AND volume spike.
Short when price breaks below Donchian(20) low AND price < 1d EMA50 AND volume spike.
Exit on opposite Donchian(10) break or EMA50 trend reversal.
Uses discrete position sizing (0.25) to minimize fee drag. Target: 12-37 trades/year.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_50 = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Calculate Donchian channels on 12h timeframe
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian(20) for entry
    donch_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donch_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Donchian(10) for exit
    donch_high_10 = pd.Series(high_12h).rolling(window=10, min_periods=10).max().values
    donch_low_10 = pd.Series(low_12h).rolling(window=10, min_periods=10).min().values
    
    # Align all 12h indicators to lower timeframe
    donch_high_20_aligned = align_htf_to_ltf(prices, df_12h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_12h, donch_low_20)
    donch_high_10_aligned = align_htf_to_ltf(prices, df_12h, donch_high_10)
    donch_low_10_aligned = align_htf_to_ltf(prices, df_12h, donch_low_10)
    
    # Volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50 and Donchian periods
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or
            np.isnan(donch_high_10_aligned[i]) or np.isnan(donch_low_10_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian(20) high AND price > 1d EMA50 AND volume spike
            if close[i] > donch_high_20_aligned[i] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian(20) low AND price < 1d EMA50 AND volume spike
            elif close[i] < donch_low_20_aligned[i] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            if position == 1:
                # Exit long: price breaks below Donchian(10) low OR price crosses below 1d EMA50
                if close[i] < donch_low_10_aligned[i] or close[i] < ema_50_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above Donchian(10) high OR price crosses above 1d EMA50
                if close[i] > donch_high_10_aligned[i] or close[i] > ema_50_aligned[i]:
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "12H_Donchian20_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0