#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and 4h volume spike confirmation.
# Long when price breaks above Donchian(20) high with price > 12h EMA50 (bullish trend) and 4h volume > 2.0x 20-period average.
# Short when price breaks below Donchian(20) low with price < 12h EMA50 (bearish trend) and 4h volume > 2.0x 20-period average.
# Exit on opposite Donchian level (low for longs, high for shorts).
# Uses 4h timeframe for optimal trade frequency (target: 75-200 total trades over 4 years) and 12h EMA50 for strong trend filter.
# Volume spike confirmation (2.0x) reduces false breakouts. Designed to work in both bull (trend continuation) and bear (mean reversion at extremes) markets.

name = "4h_Donchian20_Breakout_12hEMA50_4hVolumeSpike"
timeframe = "4h"
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
    
    # --- 4h Indicators (LTF) ---
    # Donchian Channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation: > 2.0x 20-period average (stricter filter to reduce trade frequency)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume > (2.0 * vol_ma_20)
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA(50) - trend filter (longer period for stronger trend signal)
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(high_20[i]) or
            np.isnan(low_20[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + price > 12h EMA50 (bullish) + 4h volume spike
            if (close[i] > high_20[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + price < 12h EMA50 (bearish) + 4h volume spike
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals