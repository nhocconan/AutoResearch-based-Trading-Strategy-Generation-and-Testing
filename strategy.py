#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w EMA50 trend filter and 1d volume spike confirmation.
# Long when price breaks above upper Donchian(20) with price > 1w EMA50 (bullish trend) and 1d volume > 2.0x 20-period average.
# Short when price breaks below lower Donchian(20) with price < 1w EMA50 (bearish trend) and 1d volume > 2.0x 20-period average.
# Exit on opposite Donchian level (lower for longs, upper for shorts).
# Uses 1d timeframe for low trade frequency and 1w HTF for robust trend filter. Volume spike confirms breakout strength.
# Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Donchian20_Breakout_1wEMA50_1dVolumeSpike"
timeframe = "1d"
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
    
    # --- 1d Indicators (LTF) ---
    # 1d volume confirmation: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume > (2.0 * vol_ma_20)
    
    # 1d Donchian(20) channels
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w EMA(50) - trend filter
    ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_aligned = align_htf_to_ltf(prices, df_1w, ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(ema_50_aligned[i]) or
            np.isnan(volume_spike_1d[i]) or
            np.isnan(donchian_high[i]) or
            np.isnan(donchian_low[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + price > 1w EMA50 (bullish) + 1d volume spike
            if (close[i] > donchian_high[i] and 
                close[i] > ema_50_aligned[i] and 
                volume_spike_1d[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + price < 1w EMA50 (bearish) + 1d volume spike
            elif (close[i] < donchian_low[i] and 
                  close[i] < ema_50_aligned[i] and 
                  volume_spike_1d[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals