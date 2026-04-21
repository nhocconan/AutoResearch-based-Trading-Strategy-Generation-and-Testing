#!/usr/bin/env python3
"""
4h_Donchian20_Breakout_VolumeSpike_ChopRegime_v1
Hypothesis: 4h Donchian(20) breakout with volume confirmation (>1.8x average) and choppiness regime filter (CHOP>61.8 = range, <38.2 = trend). 
In trending markets (CHOP<38.2): breakout continuation (long above upper band, short below lower band). 
In ranging markets (CHOP>61.8): mean reversion at bands (short upper band, long lower band). 
Volume confirmation filters false breakouts. ATR(14) stoploss (2.0x) and discrete sizing (0.25). 
Designed for 4h timeframe to target 75-200 trades over 4 years (19-50/year). Works in bull/bear via regime adaptation.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    # === Choppiness Index (14-period) for regime detection ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    highest_high = pd.Series(high).rolling(window=14, min_periods=14).max().values
    lowest_low = pd.Series(low).rolling(window=14, min_periods=14).min().values
    
    chop = 100 * np.log10(atr.sum(axis=0) / (highest_high - lowest_low)) / np.log10(14)
    chop = pd.Series(chop).rolling(window=14, min_periods=14).mean().values
    
    # === Donchian Channel (20-period) ===
    highest_high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # === Volume confirmation (20-period average) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(60, n):
        # Skip if indicators not ready
        if (np.isnan(chop[i]) or np.isnan(highest_high_20[i]) or np.isnan(lowest_low_20[i]) 
            or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        upper = highest_high_20[i]
        lower = lowest_low_20[i]
        vol_avg = vol_ma[i]
        chop_val = chop[i]
        
        # Volume confirmation: current volume > 1.8x average (strict filter)
        volume_confirmed = volume_now > 1.8 * vol_avg
        
        if position == 0:
            if chop_val < 38.2:  # Trending regime
                # Breakout continuation
                long_condition = (price > upper) and volume_confirmed
                short_condition = (price < lower) and volume_confirmed
                
                if long_condition:
                    signals[i] = 0.25
                    position = 1
                    entry_price = price
                elif short_condition:
                    signals[i] = -0.25
                    position = -1
                    entry_price = price
            elif chop_val > 61.8:  # Ranging regime
                # Mean reversion at bands
                long_condition = (price < lower) and volume_confirmed
                short_condition = (price > upper) and volume_confirmed
                
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
            # Regime change exit (if chop moves to extreme opposite)
            elif (position == 1 and chop_val > 61.8) or (position == -1 and chop_val < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Regime change exit (if chop moves to extreme opposite)
            elif (position == 1 and chop_val > 61.8) or (position == -1 and chop_val < 38.2):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_Donchian20_Breakout_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0