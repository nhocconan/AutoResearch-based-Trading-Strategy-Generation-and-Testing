#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ChopRegime_v2
Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.8x average) and choppiness regime filter (CHOP>61.8 = range). 
Long when price breaks above Donchian upper band in ranging markets; short when price breaks below lower band.
Mean reversion logic works in both bull and bear markets by fading extremes during consolidation.
Volume confirmation reduces false breakouts. ATR(14) stoploss (2.0x) and discrete sizing (0.25).
Designed to avoid overtrading (< 30 trades/year per symbol) via tight entry conditions.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === 4h Donchian channels (20-period) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Upper band: highest high over past 20 bars
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    # Lower band: lowest low over past 20 bars
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (50-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=50, min_periods=50).mean().values
    
    # === ATR (14-period) for stoploss ===
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Choppiness Index regime filter (14-period) ===
    # CHOP = 100 * log10(sum(TR over n) / (n * (max(high) - min(low)))) / log10(n)
    # Range: 0-100, >61.8 = ranging, <38.2 = trending
    sum_tr = pd.Series(tr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    denominator = 14 * (max_high - min_low)
    # Avoid division by zero
    chop_raw = np.where(denominator != 0, sum_tr / denominator, 1.0)
    chop = 100 * np.log10(np.maximum(chop_raw, 1e-10)) / np.log10(14)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) 
            or np.isnan(atr[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        vol_avg = vol_ma[i]
        chop_value = chop[i]
        
        # Volume confirmation: current volume > 1.8x average (tight filter)
        volume_confirmed = volume_now > 1.8 * vol_avg
        
        # Regime filter: only trade in ranging markets (CHOP > 61.8)
        ranging_market = chop_value > 61.8
        
        if position == 0:
            # Enter only in ranging markets with volume confirmation
            long_condition = (price > highest_high[i]) and volume_confirmed and ranging_market
            short_condition = (price < lowest_low[i]) and volume_confirmed and ranging_market
            
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
            # Exit when price re-enters Donchian channel (mean reversion complete)
            elif price < highest_high[i] and price > lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit when price re-enters Donchian channel (mean reversion complete)
            elif price < highest_high[i] and price > lowest_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopRegime_v2"
timeframe = "4h"
leverage = 1.0