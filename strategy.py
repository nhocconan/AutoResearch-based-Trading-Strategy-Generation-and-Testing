#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian channel breakout (20-period) with 1d EMA trend filter and volume confirmation.
# Uses 1d EMA(34) for trend bias and Donchian breakout for entry timing.
# Includes volume filter (current volume > 1.5x 20-bar SMA) to avoid low-momentum breakouts.
# Designed for low trade frequency (19-50/year) to minimize fee drag.
# Works in bull/bear: 1d EMA avoids counter-trend trades, Donchian captures breakouts with momentum.

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
    
    # === 1d Indicators: EMA(34) for trend filter ===
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h Indicators: Donchian Channel (20) ===
    high_4h = pd.Series(high)
    low_4h = pd.Series(low)
    donchian_high = high_4h.rolling(window=20, min_periods=20).max().values
    donchian_low = low_4h.rolling(window=20, min_periods=20).min().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_34_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Donchian high (20-period breakout)
        # 2. 1d price above EMA34 (bullish trend bias)
        # 3. Volume confirmation
        if (close[i] > donchian_high[i] and
            close[i] > ema_34_1d_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Donchian low (20-period breakdown)
        # 2. 1d price below EMA34 (bearish trend bias)
        # 3. Volume confirmation
        elif (close[i] < donchian_low[i] and
              close[i] < ema_34_1d_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_EMA34_VolFilter_v1"
timeframe = "4h"
leverage = 1.0