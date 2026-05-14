#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and 4h volume confirmation.
# Long when price breaks above 20-period high with 1d EMA(50) bullish (close > EMA) and 4h volume > 1.8x 20-period average.
# Short when price breaks below 20-period low with 1d EMA(50) bearish (close < EMA) and 4h volume > 1.8x 20-period average.
# Exit on opposite Donchian level (20-period low for longs, 20-period high for shorts).
# Uses discrete position sizing (0.25) to minimize fee churn and volume confirmation to reduce false breakouts.
# Target: 100-200 total trades over 4 years = 25-50/year for 4h timeframe.
# Works in bull/bear: 1d EMA ensures trend alignment, Donchian provides structure, volume confirms momentum.

name = "4h_Donchian20_Breakout_1dEMA50_4hVolumeConfirm"
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
    # 4h volume spike: > 1.8x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume > (1.8 * vol_ma_20)
    
    # Donchian channels (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
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
    
    for i in range(20, n):  # Start after Donchian warmup
        # Skip if missing data
        if (np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_spike_4h[i]) or
            np.isnan(high_20[i]) or
            np.isnan(low_20[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 20-period high + 1d EMA bullish (close > EMA) + 4h volume spike
            if (close[i] > high_20[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below 20-period low + 1d EMA bearish (close < EMA) + 4h volume spike
            elif (close[i] < low_20[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 20-period low
            if close[i] < low_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above 20-period high
            if close[i] > high_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals