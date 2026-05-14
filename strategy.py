#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout + 1d EMA50 trend + volume spike confirmation (1.5x). 
# Long when price breaks above Donchian upper with price > 1d EMA50 and volume > 1.5x 20-bar avg.
# Short when price breaks below Donchian lower with price < 1d EMA50 and volume > 1.5x 20-bar avg.
# Exit on opposite Donchian level. Uses tight volume filter (1.5x) and discrete sizing (0.25) to limit trades.
# Designed for trend continuation in bull markets and mean reversion at extremes in bear markets.
# Target: 100-180 total trades over 4 years (25-45/year) to avoid fee drag.

name = "4h_Donchian20_Breakout_1dEMA50_VolumeSpike_v1"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 4h Indicators (LTF) ---
    # Donchian Channel (20) - breakout levels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 4h volume confirmation: > 1.5x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_4h = volume > (1.5 * vol_ma_20)
    
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
        if (np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i]) or
            np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike_4h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian high + price > 1d EMA50 + volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike_4h[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low + price < 1d EMA50 + volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike_4h[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Donchian low
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above Donchian high
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals