#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA50 trend filter and 12h volume spike confirmation.
# Long when price breaks above upper Donchian channel with price > 1d EMA50 (bullish trend) and 12h volume > 2.0x 20-period average.
# Short when price breaks below lower Donchian channel with price < 1d EMA50 (bearish trend) and 12h volume > 2.0x 20-period average.
# Exit on opposite Donchian level (lower for longs, upper for shorts).
# Uses Donchian(20) for clear structure, 1d EMA50 for responsive trend filter (works in both bull/bear),
# and volume spike to confirm institutional interest. Target: 50-150 total trades over 4 years (12-37/year).

name = "12h_Donchian20_Breakout_1dEMA50_12hVolumeSpike"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 12h Indicators (LTF) ---
    # 12h Donchian channel (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 12h volume confirmation: > 2.0x 20-period average (balanced filter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_12h = volume > (2.0 * vol_ma_20)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50) - trend filter
    ema_50 = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + price > 1d EMA50 (bullish) + 12h volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike_12h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + price < 1d EMA50 (bearish) + 12h volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike_12h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals