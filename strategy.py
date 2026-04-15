#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation.
# Uses 1d EMA(50) for trend bias and 4h Donchian channels for breakout signals.
# Includes volume filter (current volume > 1.8x 20-bar SMA) to avoid false breakouts.
# Designed for low trade frequency (20-40/year) to minimize fee drag.
# Works in bull/bear: Donchian captures breakouts, 1d EMA avoids counter-trend trades, volume confirms momentum.

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
    
    # === 4h Indicators: Donchian Channels ===
    # Donchian(20): upper = max(high, 20), lower = min(low, 20)
    high_4h = pd.Series(df_4h['high'].values)
    low_4h = pd.Series(df_4h['low'].values)
    donchian_upper_4h = high_4h.rolling(window=20, min_periods=20).max().values
    donchian_lower_4h = low_4h.rolling(window=20, min_periods=20).min().values
    
    donchian_upper_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_upper_4h)
    donchian_lower_4h_aligned = align_htf_to_ltf(prices, df_4h, donchian_lower_4h)
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(50) for trend bias
    close_1d = pd.Series(df_1d['close'].values)
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.8x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_upper_4h_aligned[i]) or np.isnan(donchian_lower_4h_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian upper channel (20-period high breakout)
        # 2. 1d price above EMA50 (bullish long-term trend bias)
        # 3. Volume confirmation
        if (close[i] > donchian_upper_4h_aligned[i] and
            close[i] > ema_50_1d_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian lower channel (20-period low breakout)
        # 2. 1d price below EMA50 (bearish long-term trend bias)
        # 3. Volume confirmation
        elif (close[i] < donchian_lower_4h_aligned[i] and
              close[i] < ema_50_1d_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_EMA50_VolumeFilter_v1"
timeframe = "4h"
leverage = 1.0