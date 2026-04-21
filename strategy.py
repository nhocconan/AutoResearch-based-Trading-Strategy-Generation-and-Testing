#!/usr/bin/env python3
"""
6h_Donchian_Breakout_WeeklyPivot_Direction_Volume
Hypothesis: On 6h timeframe, take long when price breaks above 20-period Donchian high + weekly pivot (from 1w) indicates bullish bias + volume confirmation (>1.5x avg). Take short when price breaks below 20-period Donchian low + weekly pivot indicates bearish bias + volume confirmation. Weekly pivot acts as a regime filter to avoid counter-trend trades. Works in bull/bear by aligning with higher timeframe bias. Target 12-37 trades/year on 6h.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # Load weekly HTF data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # === Weekly pivot point (classic) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    pp_1w = (high_1w + low_1w + close_1w) / 3.0
    pp_1w_aligned = align_htf_to_ltf(prices, df_1w, pp_1w)
    
    # === 6h Donchian channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    donch_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation: 20-period volume average on 6h ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = np.where(vol_ma_20 != 0, volume / vol_ma_20, 1.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(60, n):  # Start after Donchian warmup
        # Skip if indicators not ready
        if (np.isnan(donch_high[i]) or
            np.isnan(donch_low[i]) or
            np.isnan(pp_1w_aligned[i]) or
            np.isnan(vol_ratio[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        weekly_pivot = pp_1w_aligned[i]
        vol_spike = vol_ratio[i]
        
        if position == 0:
            # Long: Price breaks above Donchian high + weekly pivot bias bullish (price > pivot) + volume spike
            if (price_close > donch_high[i] and 
                price_close > weekly_pivot and 
                vol_spike > 1.5):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below Donchian low + weekly pivot bias bearish (price < pivot) + volume spike
            elif (price_close < donch_low[i] and 
                  price_close < weekly_pivot and 
                  vol_spike > 1.5):
                signals[i] = -0.25
                position = -1
        
        elif position != 0:
            # Exit when price returns to the opposite Donchian level (mean reversion within channel)
            if position == 1 and price_close < donch_low[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and price_close > donch_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Hold position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "6h_Donchian_Breakout_WeeklyPivot_Direction_Volume"
timeframe = "6h"
leverage = 1.0