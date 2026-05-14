#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and 4h volume spike filter.
# Long when price breaks above upper Donchian(20) with 1d EMA(50) bullish (close > EMA) and 4h volume > 2.0x 20-period average.
# Short when price breaks below lower Donchian(20) with 1d EMA(50) bearish (close < EMA) and 4h volume > 2.0x 20-period average.
# Exit on opposite Donchian level (lower for longs, upper for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn and volume spike filter to reduce false breakouts.
# Target: 75-200 total trades over 4 years = 19-50/year for 4h timeframe.
# Works in bull/bear: 1d EMA ensures trend alignment, Donchian provides clear structure, volume spike confirms momentum.

name = "4h_Donchian20_Breakout_1dEMA50_4hVolumeSpike"
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
    # 4h volume spike: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume > (2.0 * vol_ma_20)
    
    # Donchian(20) channels
    upper_dc = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lower_dc = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(50)
    ema_50_1d = pd.Series(close_1d).ewm(span=50, min_periods=50, adjust=False).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_spike_4h[i]) or
            np.isnan(upper_dc[i]) or
            np.isnan(lower_dc[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + 1d EMA bullish + 4h volume spike
            if (close[i] > upper_dc[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + 1d EMA bearish + 4h volume spike
            elif (close[i] < lower_dc[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian
            if close[i] < lower_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian
            if close[i] > upper_dc[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals