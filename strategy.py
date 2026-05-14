#!/usr/bin/env python3
# Hypothesis: 6h Donchian(20) breakout with weekly trend filter (price > weekly EMA50) and 6h volume confirmation (1.5x 20-period average).
# Long when price breaks above Donchian upper band with price > weekly EMA50 and volume > 1.5x average.
# Short when price breaks below Donchian lower band with price < weekly EMA50 and volume > 1.5x average.
# Exit on opposite Donchian band (lower for longs, upper for shorts).
# Uses weekly HTF for major trend to avoid counter-trend trades in bear markets.
# Volume confirmation reduces false breakouts. Target: 50-150 total trades over 4 years (12-37/year).

name = "6h_Donchian20_Breakout_WeeklyEMA50_VolumeConfirm_v1"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # --- 6h Indicators (LTF) ---
    # 6h Donchian channels (20-period)
    high_roll = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_roll = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_upper = high_roll
    donchian_lower = low_roll
    
    # 6h volume confirmation: > 1.5x 20-period average (tight filter)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (1.5 * vol_ma_20)
    
    # --- Weekly Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # Weekly EMA(50) - major trend filter
    weekly_ema_50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema_50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema_50)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i]) or
            np.isnan(volume_spike[i]) or
            np.isnan(weekly_ema_50_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above Donchian upper + price > weekly EMA50 (bullish trend) + volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > weekly_ema_50_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian lower + price < weekly EMA50 (bearish trend) + volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < weekly_ema_50_aligned[i] and 
                  volume_spike[i]):
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