#!/usr/bin/env python3
# Hypothesis: 1h 4-hour Donchian breakout with 1d EMA(34) trend filter and 1h volume spike filter.
# Long when price breaks above 4h Donchian upper (20) with 1d EMA bullish and 1h volume > 2.0x 20-period average.
# Short when price breaks below 4h Donchian lower (20) with 1d EMA bearish and 1h volume > 2.0x 20-period average.
# Exit on opposite Donchian level (lower for longs, upper for shorts).
# Uses 4h/1d for signal direction (HTF), 1h only for entry timing and volume confirmation.
# Session filter: 08-20 UTC to avoid low-liquidity hours.
# Discrete position sizing: 0.20 to minimize fee churn and manage drawdown.
# Target: 60-150 total trades over 4 years = 15-37/year for 1h timeframe.

name = "1h_Donchian20_Breakout_1dEMA34_1hVolumeSpike"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Precompute session hours (08-20 UTC)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    open_ = prices['open'].values
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 1h Indicators (LTF) ---
    # 1h volume spike: > 2.0x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_1h = volume > (2.0 * vol_ma_20)
    
    # --- 4h Indicators (HTF) ---
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # 4h Donchian Channel (20)
    upper_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    lower_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    upper_4h_aligned = align_htf_to_ltf(prices, df_4h, upper_4h)
    lower_4h_aligned = align_htf_to_ltf(prices, df_4h, lower_4h)
    
    # --- 1d Indicators (HTF) ---
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    close_1d = df_1d['close'].values
    
    # 1d EMA(34)
    ema_34_1d = pd.Series(close_1d).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if outside session or missing data
        if not in_session[i] or \
           (np.isnan(ema_34_1d_aligned[i]) or
            np.isnan(volume_spike_1h[i]) or
            np.isnan(upper_4h_aligned[i]) or
            np.isnan(lower_4h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above 4h Donchian upper + 1d EMA bullish + 1h volume spike
            if (close[i] > upper_4h_aligned[i] and 
                close[i] > ema_34_1d_aligned[i] and 
                volume_spike_1h[i]):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below 4h Donchian lower + 1d EMA bearish + 1h volume spike
            elif (close[i] < lower_4h_aligned[i] and 
                  close[i] < ema_34_1d_aligned[i] and 
                  volume_spike_1h[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below 4h Donchian lower
            if close[i] < lower_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above 4h Donchian upper
            if close[i] > upper_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals