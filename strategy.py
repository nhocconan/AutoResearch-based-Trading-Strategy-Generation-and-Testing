#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Uses 1d EMA34 for trend bias and 12h Donchian channels for breakout entries.
# Includes volume filter (current volume > 1.3x 20-bar SMA) to avoid low-momentum breakouts.
# Designed for low trade frequency (12-25/year) to minimize fee drag.
# Works in bull/bear: Donchian captures breakouts, EMA34 avoids counter-trend trades, volume confirms momentum.

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
    if len(df_12h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 12h Indicators: Donchian Channels ===
    high_12h = pd.Series(df_12h['high'].values)
    low_12h = pd.Series(df_12h['low'].values)
    donchian_high_20 = high_12h.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_12h.rolling(window=20, min_periods=20).min().values
    
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_12h, donchian_low_20)
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(34) for trend bias
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current 12h volume > 1.3x 20-period 12h volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 12h Donchian High(20)
        # 2. 1d price above EMA34 (bullish long-term trend bias)
        # 3. Volume confirmation
        if (close[i] > donchian_high_20_aligned[i] and
            close[i] > ema_34_1d_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 12h Donchian Low(20)
        # 2. 1d price below EMA34 (bearish long-term trend bias)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_20_aligned[i] and
              close[i] < ema_34_1d_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Donchian20_EMA34_VolFilter_v1"
timeframe = "12h"
leverage = 1.0