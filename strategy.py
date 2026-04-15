#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Donchian breakout (20) + 1d trend filter (EMA200) + volume confirmation + session filter (08-20 UTC).
# Uses 4h for signal direction (Donchian breakout with volume) and 1d EMA200 for trend bias.
# 1h is used only for entry timing precision. Designed for low trade frequency (15-35/year) to minimize fee drag.
# Works in bull/bear: Donchian captures breakouts, volume confirms, 1d EMA avoids counter-trend trades, session filter reduces noise.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian Channel (20) ===
    high_4h = pd.Series(df_4h['high'].values)
    low_4h = pd.Series(df_4h['low'].values)
    donchian_high_20 = high_4h.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_4h.rolling(window=20, min_periods=20).min().values
    
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(200) for trend bias
    ema_200_1d = pd.Series(df_1d['close'].values).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 1h Indicators: Volume Filter ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC (precomputed for efficiency)
    hours = prices.index.hour
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip outside session
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian high (20)
        # 2. 1d price above EMA200 (bullish long-term trend bias)
        # 3. Volume confirmation
        if (close[i] > donchian_high_20_aligned[i] and
            close[i] > ema_200_1d_aligned[i] and
            vol_confirm):
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian low (20)
        # 2. 1d price below EMA200 (bearish long-term trend bias)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_20_aligned[i] and
              close[i] < ema_200_1d_aligned[i] and
              vol_confirm):
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Donchian20_EMA200_Volume_SessionFilter_v1"
timeframe = "1h"
leverage = 1.0