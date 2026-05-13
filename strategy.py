#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1w EMA50 trend filter and 1d volume spike confirmation.
# Long when price breaks above 20-period 12h Donchian upper band AND price > 1w EMA50 AND 1d volume > 2.0 * 20-period average volume.
# Short when price breaks below 20-period 12h Donchian lower band AND price < 1w EMA50 AND 1d volume > 2.0 * 20-period average volume.
# Exit when price crosses the 12-period 12h Donchian middle band (mean reversion to midpoint).
# Uses discrete position sizing (0.25) to limit fee churn. Designed for BTC/ETH robustness by capturing strong trends with volume confirmation in both bull and bear markets.
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe.

name = "12h_Donchian20_Breakout_1wEMA50_1dVolumeSpike_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 12h Donchian(20) channels
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Upper band: 20-period high
    donchian_20_upper = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    # Lower band: 20-period low
    donchian_20_lower = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    # Middle band: 12-period average of upper and lower (faster exit)
    donchian_12_middle = (pd.Series(high_12h).rolling(window=12, min_periods=12).mean().values + 
                          pd.Series(low_12h).rolling(window=12, min_periods=12).mean().values) / 2
    
    # Align Donchian bands to 12h timeframe
    donchian_20_upper_aligned = align_htf_to_ltf(prices, df_12h, donchian_20_upper)
    donchian_20_lower_aligned = align_htf_to_ltf(prices, df_12h, donchian_20_lower)
    donchian_12_middle_aligned = align_htf_to_ltf(prices, df_12h, donchian_12_middle)
    
    # Calculate 1w EMA50 for trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Calculate 1d volume spike filter (HTF)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    vol_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_1d > (2.0 * vol_ma_20)  # Threshold increased to 2.0x to reduce trades
    volume_spike_aligned = align_htf_to_ltf(prices, df_1d, volume_spike.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if any required data is NaN
        if (np.isnan(donchian_20_upper_aligned[i]) or 
            np.isnan(donchian_20_lower_aligned[i]) or
            np.isnan(donchian_12_middle_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper band AND price > 1w EMA50 AND volume spike
            if (close[i] > donchian_20_upper_aligned[i] and 
                close[i] > ema_50_1w_aligned[i] and 
                volume_spike_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower band AND price < 1w EMA50 AND volume spike
            elif (close[i] < donchian_20_lower_aligned[i] and 
                  close[i] < ema_50_1w_aligned[i] and 
                  volume_spike_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses below Donchian middle band
            if close[i] < donchian_12_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses above Donchian middle band
            if close[i] > donchian_12_middle_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals