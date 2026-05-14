#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with 12h EMA(50) trend filter and 6h volume spike filter.
# Long when price breaks above Donchian upper band with 12h EMA bullish (close > EMA) and 6h volume > 1.8x 20-period average.
# Short when price breaks below Donchian lower band with 12h EMA bearish (close < EMA) and 6h volume > 1.8x 20-period average.
# Exit on opposite Donchian band (lower for longs, upper for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn and volume spike filter to reduce false breakouts.
# Target: 50-150 total trades over 4 years = 12-37/year for 6h timeframe.
# Works in bull/bear: 12h EMA ensures trend alignment, Donchian provides structure, volume spike confirms momentum.

name = "6h_Donchian20_Breakout_12hEMA50_6hVolumeSpike"
timeframe = "6h"
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
    
    # --- 6h Indicators (LTF) ---
    # 6h Donchian bands (20-period)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 6h volume spike: > 1.8x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_6h = volume > (1.8 * vol_ma_20)
    
    # --- 12h Indicators (HTF) ---
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 60:
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # 12h EMA(50)
    ema_50_12h = pd.Series(close_12h).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_50_12h_aligned[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(volume_spike_6h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper + 12h EMA bullish (close > EMA) + 6h volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_12h_aligned[i] and 
                volume_spike_6h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + 12h EMA bearish (close < EMA) + 6h volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_12h_aligned[i] and 
                  volume_spike_6h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian lower
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian upper
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals