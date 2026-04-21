#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ChopRegime
Hypothesis: Donchian(20) breakout on 4h with volume confirmation (>1.8x 20-period MA) and chop regime filter (CHOP(14) > 61.8 = range, < 38.2 = trending). 
In trending regime (CHOP < 38.2): breakout signals only. In ranging regime (CHOP > 61.8): mean reversion at Donchian bands.
Uses ATR(14) stoploss (2.0x) and discrete sizing (0.25). Target: 80-140 total trades (20-35/year) to balance edge and fee drag.
Works in both bull and bear via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for higher timeframe context)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d < 40):
        return np.zeros(n)
    
    # === 4h Donchian(20) channels ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    donchian_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    donchian_mid = (donchian_high + donchian_low) / 2
    
    # === 4h ATR(14) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume spike filter (1.8x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === Choppiness Index (CHOP) regime filter ===
    # CHOP = 100 * log10(sum(ATR(14) over n) / (log10(n) * (max(high,n) - min(low,n))))
    # Simplified: CHOP > 61.8 = ranging, CHOP < 38.2 = trending
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / (np.log10(14) * (max_high_14 - min_low_14)))
    # Handle division by zero or invalid values
    chop = np.where((max_high_14 - min_low_14) > 0, chop, 50.0)  # neutral when no range
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) 
            or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        vol_avg = vol_ma[i]
        upper = donchian_high[i]
        lower = donchian_low[i]
        mid = donchian_mid[i]
        chop_val = chop[i]
        
        # Volume spike: current volume > 1.8x average
        volume_spike = volume_now > 1.8 * vol_avg
        
        if position == 0:
            # Regime-based entry logic
            if chop_val < 38.2:  # Trending regime: breakout only
                long_condition = (price > upper) and volume_spike
                short_condition = (price < lower) and volume_spike
            elif chop_val > 61.8:  # Ranging regime: mean reversion at bands
                long_condition = (price < lower) and (price > mid * 0.999) and volume_spike  # near lower band
                short_condition = (price > upper) and (price < mid * 1.001) and volume_spike  # near upper band
            else:  # Transition regime: no entries
                long_condition = False
                short_condition = False
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: opposite band touch or regime shift to ranging
            elif price < lower or (chop_val > 61.8 and price < mid):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: opposite band touch or regime shift to ranging
            elif price > upper or (chop_val > 61.8 and price > mid):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0