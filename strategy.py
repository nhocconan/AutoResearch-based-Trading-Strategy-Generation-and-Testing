#!/usr/bin/env python3
"""
6h_ElderRay_Regime_VolumeFilter
Hypothesis: 6h Elder Ray (Bull/Bear Power) with 1d trend regime (EMA50) and volume confirmation (>1.5x 20-period MA).
Long when Bull Power > 0, price above 1d EMA50, and volume > 1.5x average.
Short when Bear Power < 0, price below 1d EMA50, and volume > 1.5x average.
Elder Ray measures bull/bear strength relative to EMA13, filtering weak moves.
Volume confirmation ensures participation. 6h timeframe reduces noise vs lower TFs.
Target: 50-150 total trades over 4 years (12-37/year) with discrete sizing (0.25) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d EMA50 for trend regime ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h EMA13 for Elder Ray calculation ===
    close = prices['close'].values
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # === Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13 ===
    high = prices['high'].values
    low = prices['low'].values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # === Volume confirmation (1.5x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(ema_13[i]) or 
            np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        ema_50_1d_val = ema_50_1d_aligned[i]
        vol_avg = vol_ma[i]
        bull_val = bull_power[i]
        bear_val = bear_power[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: Bull Power > 0, price above 1d EMA50, volume confirm
            long_condition = (bull_val > 0) and (price > ema_50_1d_val) and volume_confirm
            # Short: Bear Power < 0, price below 1d EMA50, volume confirm
            short_condition = (bear_val < 0) and (price < ema_50_1d_val) and volume_confirm
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Exit conditions
            if position == 1:
                # Exit long: Bull Power turns negative OR price crosses below 1d EMA50
                if (bull_val <= 0) or (price < ema_50_1d_val):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                # Exit short: Bear Power turns positive OR price crosses above 1d EMA50
                if (bear_val >= 0) or (price > ema_50_1d_val):
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_Regime_VolumeFilter"
timeframe = "6h"
leverage = 1.0