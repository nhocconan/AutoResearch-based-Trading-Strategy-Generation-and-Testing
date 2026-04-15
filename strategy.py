#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20) with 12h EMA trend filter and volume confirmation.
# Uses 12h EMA34 for trend bias and Donchian channels (20-period high/low) on 4h for breakout entries.
# Includes volume filter (current volume > 1.3x 20-bar SMA) to avoid low-momentum breakouts.
# Designed for low trade frequency (20-40/year) to minimize fee drag while capturing strong trends.
# Works in bull/bear: EMA34 filter avoids counter-trend trades, Donchian breakouts catch momentum.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h and 12h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_12h = get_htf_data(prices, '12h')
    if len(df_4h) < 50 or len(df_12h) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: Donchian Channels (20) ===
    high_4h = pd.Series(df_4h['high'].values)
    low_4h = pd.Series(df_4h['low'].values)
    donchian_high_20 = high_4h.rolling(window=20, min_periods=20).max().values
    donchian_low_20 = low_4h.rolling(window=20, min_periods=20).min().values
    
    donchian_high_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_20)
    donchian_low_20_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_20)
    
    # === 12h Indicators: EMA34 Trend Filter ===
    close_12h = pd.Series(df_12h['close'].values)
    ema_34_12h = close_12h.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current 4h volume > 1.3x 20-period 4h volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_20_aligned[i]) or np.isnan(donchian_low_20_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 20-period Donchian high
        # 2. 12h price above EMA34 (bullish trend bias)
        # 3. Volume confirmation
        if (close[i] > donchian_high_20_aligned[i] and
            close[i] > ema_34_12h_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 20-period Donchian low
        # 2. 12h price below EMA34 (bearish trend bias)
        # 3. Volume confirmation
        elif (close[i] < donchian_low_20_aligned[i] and
              close[i] < ema_34_12h_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_EMA34_VolFilter_v1"
timeframe = "4h"
leverage = 1.0