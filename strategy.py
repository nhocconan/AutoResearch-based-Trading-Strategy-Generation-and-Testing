#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and 4h volume spike confirmation (>2.0x 20-bar avg).
# Long when price breaks above upper Donchian(20) with price > 1d EMA50 (bullish) and 4h volume > 2.0x 20-period average.
# Short when price breaks below lower Donchian(20) with price < 1d EMA50 (bearish) and 4h volume > 2.0x 20-period average.
# Exit on opposite Donchian level (lower for longs, upper for shorts).
# Uses 1d HTF for trend to reduce noise and overtrading. Volume spike confirmation reduces false breakouts.
# Target: 75-200 total trades over 4 years (19-50/year) to stay within fee drag limits.
# Works in both bull (trend-following breakouts) and bear (mean-reversion via opposite breakouts in ranging markets).

name = "4h_Donchian20_Breakout_1dEMA50_4hVolumeSpike_v1"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF) ---
    # 4h volume confirmation: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume > (2.0 * vol_ma_20)
    
    # Donchian(20) channels
    upper_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike_4h[i]) or
            np.isnan(upper_20[i]) or
            np.isnan(lower_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian(20) + price > 1d EMA50 (bullish) + 4h volume spike
            if (close[i] > upper_20[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian(20) + price < 1d EMA50 (bearish) + 4h volume spike
            elif (close[i] < lower_20[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian(20)
            if close[i] < lower_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian(20)
            if close[i] > upper_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals