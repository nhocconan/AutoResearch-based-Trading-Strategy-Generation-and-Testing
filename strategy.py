#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20) with 1d EMA(50) trend filter and volume confirmation.
# Uses 1d EMA(50) for trend bias and Donchian breakout for entry timing.
# Volume filter requires current volume > 1.8x 20-bar SMA to avoid low-momentum breakouts.
# Designed for low trade frequency (20-50/year) to minimize fee drag. Works in bull/bear:
# - Bull: EMA(50) up, buy Donchian breakouts
# - Bear: EMA(50) down, sell Donchian breakdowns
# - Chop: EMA(50) flat, fewer trades as price stays inside channel

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian Channel (20) ===
    high_4h = pd.Series(high)
    low_4h = pd.Series(low)
    donchian_high = high_4h.rolling(window=20, min_periods=20).max().values
    donchian_low = low_4h.rolling(window=20, min_periods=20).min().values
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(50) for trend bias
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Donchian levels from 1d for additional structure (optional)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    donchian_high_1d = high_1d.rolling(window=20, min_periods=20).max().values
    donchian_low_1d = low_1d.rolling(window=20, min_periods=20).min().values
    donchian_high_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_high_1d)
    donchian_low_1d_aligned = align_htf_to_ltf(prices, df_1d, donchian_low_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.8x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(donchian_high_1d_aligned[i]) or np.isnan(donchian_low_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian high (20-period breakout)
        # 2. 1d price above EMA50 (bullish trend bias)
        # 3. Volume confirmation
        # 4. Price above 1d Donchian mid-point (avoid buying too low in range)
        if (close[i] > donchian_high[i] and
            close[i] > ema_50_1d_aligned[i] and
            vol_confirm and
            close[i] > (donchian_high_1d_aligned[i] + donchian_low_1d_aligned[i]) / 2):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian low (20-period breakdown)
        # 2. 1d price below EMA50 (bearish trend bias)
        # 3. Volume confirmation
        # 4. Price below 1d Donchian mid-point (avoid selling too high in range)
        elif (close[i] < donchian_low[i] and
              close[i] < ema_50_1d_aligned[i] and
              vol_confirm and
              close[i] < (donchian_high_1d_aligned[i] + donchian_low_1d_aligned[i]) / 2):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_EMA50_VolFilter_v1"
timeframe = "4h"
leverage = 1.0