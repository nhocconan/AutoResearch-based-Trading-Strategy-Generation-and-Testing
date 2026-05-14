#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA(50) trend filter and 4h volume spike filter.
# Long when price breaks above Donchian upper band with 1d EMA(50) bullish and 4h volume > 1.5x 20-period average.
# Short when price breaks below Donchian lower band with 1d EMA(50) bearish and 4h volume > 1.5x 20-period average.
# Exit on opposite Donchian band.
# Uses discrete position sizing (0.25) to minimize fee churn and volume spike filter to reduce false breakouts.
# Target: 75-150 total trades over 4 years = 19-37/year for 4h timeframe.
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
    # Donchian(20)
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume spike: > 1.5x 20-period average (moderate filter to balance trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume > (1.5 * vol_ma_20)
    
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
    
    for i in range(20, n):
        # Skip if missing data
        if (np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(volume_spike_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper + 1d EMA bullish (close > EMA) + 4h volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > ema_50_1d_aligned[i] and 
                volume_spike_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + 1d EMA bearish (close < EMA) + 4h volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < ema_50_1d_aligned[i] and 
                  volume_spike_4h[i]):
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