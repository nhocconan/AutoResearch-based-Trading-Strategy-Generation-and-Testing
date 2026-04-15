#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d EMA trend filter and volume confirmation.
# Uses 1d EMA200 for long-term trend bias and 12h Donchian(20) breakouts for entries.
# Volume filter ensures breakouts occur with above-average participation.
# Designed for low trade frequency (15-30/year) to minimize fee drag.
# Works in bull markets (breakouts with trend) and bear markets (breakouts against trend filtered by EMA200).

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h and 1d HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Donchian Channel (20) ===
    high_12h = pd.Series(df_12h['high'].values)
    low_12h = pd.Series(df_12h['low'].values)
    donchian_high_20 = high_12h.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_12h.rolling(window=20, min_periods=20).min().values
    
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_20)
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(200) for long-term trend bias
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === Volume Filter (12h) ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 12h Donchian high (20)
        # 2. 1d price above EMA200 (bullish long-term trend)
        # 3. Volume confirmation
        if (close[i] > donchian_high_20_aligned[i] and
            close[i] > ema_200_1d_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 12h Donchian low (20)
        # 2. 1d price below EMA200 (bearish long-term trend)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_20_aligned[i] and
              close[i] < ema_200_1d_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_EMA200_VolFilter_v1"
timeframe = "12h"
leverage = 1.0