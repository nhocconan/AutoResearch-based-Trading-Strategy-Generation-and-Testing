#!/usr/bin/env python3
"""
4h_WilliamsAlligator_ElderRay_HTFTrend_v1
Hypothesis: On 4h timeframe, combine Williams Alligator (trend direction) with Elder Ray (bull/bear power) and 1d EMA34 trend filter to capture strong moves while avoiding whipsaw. 
In bull regime (1d close > EMA34), favor longs when Alligator is bullish (jaw < teeth < lips) and Elder Bull Power > 0. 
In bear regime (1d close < EMA34), favor shorts when Alligator is bearish (jaw > teeth > lips) and Elder Bear Power < 0.
Volume confirmation (volume > 1.5x 20-period average) ensures institutional participation. Discrete sizing (0.25) minimizes fee churn. Target: 75-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA34 trend regime)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA34 for daily trend regime ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # === 4h Williams Alligator (13,8,5 SMAs smoothed by 8,5,3) ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    
    # Jaw (13-period SMA smoothed by 8)
    jaw_raw = pd.Series(close).rolling(window=13, min_periods=13).mean().values
    jaw = pd.Series(jaw_raw).rolling(window=8, min_periods=8).mean().values
    
    # Teeth (8-period SMA smoothed by 5)
    teeth_raw = pd.Series(close).rolling(window=8, min_periods=8).mean().values
    teeth = pd.Series(teeth_raw).rolling(window=5, min_periods=5).mean().values
    
    # Lips (5-period SMA smoothed by 3)
    lips_raw = pd.Series(close).rolling(window=5, min_periods=5).mean().values
    lips = pd.Series(lips_raw).rolling(window=3, min_periods=3).mean().values
    
    # === 4h Elder Ray (Bull Power = High - EMA13, Bear Power = Low - EMA13) ===
    ema_13 = pd.Series(close).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high - ema_13
    bear_power = low - ema_13
    
    # === 4h volume confirmation (volume > 1.5x 20-period average) ===
    volume = prices['volume'].values
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirmed = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    max_hold_bars = 6  # max 1 day (6 * 4h = 24h)
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(ema_34_1d_aligned[i]) or np.isnan(jaw[i]) or np.isnan(teeth[i]) or 
            np.isnan(lips[i]) or np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_confirmed[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        daily_ema = ema_34_1d_aligned[i]
        
        # Alligator conditions
        alligator_bull = (jaw[i] < teeth[i]) and (teeth[i] < lips[i])  # jaw < teeth < lips
        alligator_bear = (jaw[i] > teeth[i]) and (teeth[i] > lips[i])  # jaw > teeth > lips
        
        # Elder Ray conditions
        elder_bull = bull_power[i] > 0
        elder_bear = bear_power[i] < 0
        
        # Daily trend regime
        is_bull = price > daily_ema
        is_bear = price < daily_ema
        
        if position == 0:
            if is_bull:
                # Bull regime: long when Alligator bullish and Elder Bull Power > 0
                long_condition = alligator_bull and elder_bull and volume_confirmed
            else:  # bear regime
                # Bear regime: short when Alligator bearish and Elder Bear Power < 0
                short_condition = alligator_bear and elder_bear and volume_confirmed
            
            if is_bull and long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif is_bear and short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Time-based exit
            if bars_since_entry >= max_hold_bars:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_WilliamsAlligator_ElderRay_HTFTrend_v1"
timeframe = "4h"
leverage = 1.0