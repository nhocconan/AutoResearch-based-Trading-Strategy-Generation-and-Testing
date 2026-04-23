#!/usr/bin/env python3
"""
Hypothesis: 12h Donchian(20) breakout + 1d EMA50 trend filter + volume confirmation.
Long when price breaks above 20-bar Donchian high AND above 1d EMA50 AND volume spike.
Short when price breaks below 20-bar Donchian low AND below 1d EMA50 AND volume spike.
Exit when price crosses back through the 10-bar Donchian midpoint (mean reversion) OR opposite breakout occurs.
Uses discrete sizing 0.25 to limit fee drag. Designed for 12h timeframe to maintain 12-37 trades/year.
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
    
    # Calculate 12h Donchian channels (20-period)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Donchian high (20-period rolling max)
    donch_high_20 = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Donchian low (20-period rolling min)
    donch_low_20 = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Donchian midpoint (10-period average of high/low for exit)
    donch_mid_10 = (pd.Series(high_12h).rolling(window=10, min_periods=10).mean().values + 
                    pd.Series(low_12h).rolling(window=10, min_periods=10).mean().values) / 2
    
    # Align HTF indicators to LTF
    donch_high_20_aligned = align_htf_to_ltf(prices, df_12h, donch_high_20)
    donch_low_20_aligned = align_htf_to_ltf(prices, df_12h, donch_low_20)
    donch_mid_10_aligned = align_htf_to_ltf(prices, df_12h, donch_mid_10)
    
    # Calculate volume spike: current volume > 2.0x 20-period MA
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > 2.0 * vol_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(50, 20)  # need EMA50 and Donchian20
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(vol_ma_20[i]) or 
            np.isnan(ema_50_aligned[i]) or 
            np.isnan(donch_high_20_aligned[i]) or np.isnan(donch_low_20_aligned[i]) or np.isnan(donch_mid_10_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high AND above 1d EMA50 AND volume spike
            if close[i] > donch_high_20_aligned[i] and close[i] > ema_50_aligned[i] and volume_spike[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low AND below 1d EMA50 AND volume spike
            elif close[i] < donch_low_20_aligned[i] and close[i] < ema_50_aligned[i] and volume_spike[i]:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions
            exit_signal = False
            if position == 1:
                # Exit long: price crosses below Donchian midpoint (mean reversion) OR opposite breakout
                if close[i] < donch_mid_10_aligned[i] or close[i] < donch_low_20_aligned[i]:
                    exit_signal = True
            elif position == -1:
                # Exit short: price crosses above Donchian midpoint OR opposite breakout
                if close[i] > donch_mid_10_aligned[i] or close[i] > donch_high_20_aligned[i]:
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