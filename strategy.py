#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w HMA(21) trend filter and 1d volume spike confirmation.
# Long when price breaks above upper Donchian channel with price > 1w HMA21 (bullish trend) and 1d volume > 2.0x 20-period average.
# Short when price breaks below lower Donchian channel with price < 1w HMA21 (bearish trend) and 1d volume > 2.0x 20-period average.
# Exit on opposite Donchian level (lower for longs, upper for shorts).
# Uses 1w HTF for trend to reduce noise and overtrading vs lower TFs. Volume spike confirmation (2.0x) reduces false breakouts.
# Target: 30-100 total trades over 4 years (7-25/year) to stay within fee drag limits.

name = "1d_Donchian20_Breakout_1wHMA21_1dVolumeSpike_v1"
timeframe = "1d"
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
    
    # --- 1d Indicators (LTF) ---
    # 1d volume confirmation: > 2.0x 20-period average (tight filter to reduce trades)
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume > (2.0 * vol_ma_20)
    
    # --- 1w Indicators (HTF) ---
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 21:
        return np.zeros(n)
    close_1w = df_1w['close'].values
    
    # 1w HMA(21) - trend filter (Hull Moving Average for smooth trend)
    # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
    half_n = 21 // 2
    sqrt_n = int(np.sqrt(21))
    wma_half = pd.Series(close_1w).rolling(window=half_n, min_periods=half_n).mean().values
    wma_full = pd.Series(close_1w).rolling(window=21, min_periods=21).mean().values
    raw_hma = 2 * wma_half - wma_full
    hma_21 = pd.Series(raw_hma).rolling(window=sqrt_n, min_periods=sqrt_n).mean().values
    hma_21_aligned = align_htf_to_ltf(prices, df_1w, hma_21)
    
    # --- 1d Donchian Channel (20-period) ---
    donchian_upper = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_lower = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(1, n):
        # Skip if missing data
        if (np.isnan(hma_21_aligned[i]) or
            np.isnan(volume_spike_1d[i]) or
            np.isnan(donchian_upper[i]) or
            np.isnan(donchian_lower[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Price breaks above upper Donchian + price > 1w HMA21 (bullish) + 1d volume spike
            if (close[i] > donchian_upper[i] and 
                close[i] > hma_21_aligned[i] and 
                volume_spike_1d[i]):
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below lower Donchian + price < 1w HMA21 (bearish) + 1d volume spike
            elif (close[i] < donchian_lower[i] and 
                  close[i] < hma_21_aligned[i] and 
                  volume_spike_1d[i]):
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below lower Donchian
            if close[i] < donchian_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price breaks above upper Donchian
            if close[i] > donchian_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals