#!/usr/bin/env python3
# Hypothesis: 4h price breaks of 12h Donchian channels with volume confirmation and 1d EMA trend filter.
# Uses 12h Donchian(20) for structure, volume spike (>1.5x avg volume) for conviction, and 1d EMA50 for trend.
# Enters long when price breaks above 12h Donchian high with volume confirmation and 1d EMA50 uptrend.
# Enters short when price breaks below 12h Donchian low with volume confirmation and 1d EMA50 downtrend.
# Exits on opposite Donchian break or trend reversal. Target: 50-150 trades over 4 years with size 0.25.
# Works in bull/bear: Donchian breakouts capture trends, volume filters avoid false breaks, EMA50 aligns with higher timeframe trend.

name = "4h_Donchian20_Volume_EMA50_Trend"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 12h data for Donchian channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h Donchian(20)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (waits for 12h bar close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_12h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_12h, donchian_low)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    ema_50 = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # Volume confirmation: current volume > 1.5x 20-period average volume
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Need enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price above 12h Donchian high + volume spike + 1d EMA50 uptrend
            if (close[i] > donchian_high_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema_50_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price below 12h Donchian low + volume spike + 1d EMA50 downtrend
            elif (close[i] < donchian_low_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema_50_aligned[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price below 12h Donchian low OR 1d EMA50 downtrend
            if (close[i] < donchian_low_aligned[i]) or (close[i] < ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price above 12h Donchian high OR 1d EMA50 uptrend
            if (close[i] > donchian_high_aligned[i]) or (close[i] > ema_50_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals