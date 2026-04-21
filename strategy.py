#!/usr/bin/env python3
"""
1h_4h_1d_VolumeSpike_HTFTrendRegime_V1
Hypothesis: 1h volume spike breakouts aligned with 4h/1d trend regime. 
Volume spikes (>2.0x 20-period MA) indicate institutional interest. 
4h EMA20 and 1d EMA50 provide multi-timeframe trend confirmation. 
Only trade in direction of HTF trend to avoid whipsaws. 
Session filter (08-20 UTC) reduces noise. Target 15-37 trades/year (60-150 total over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours (08-20 UTC) once before loop
    hours = prices.index.hour  # prices.index is DatetimeIndex
    
    # Load HTF data ONCE before loop (4h for trend, 1d for regime filter)
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_4h) < 30 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h EMA20 for trend filter ===
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # === 1d EMA50 for regime filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1h Indicators (primary timeframe) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) 
            or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC only
        hour = hours[i]
        in_session = (8 <= hour <= 20)
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        vol = volume[i]
        vol_spike = vol > 2.0 * vol_ma[i]  # volume spike >2x MA
        
        # Determine HTF trend alignment
        # Long regime: price above both 4h EMA20 and 1d EMA50
        long_regime = price > ema_20_4h_aligned[i] and price > ema_50_1d_aligned[i]
        # Short regime: price below both 4h EMA20 and 1d EMA50
        short_regime = price < ema_20_4h_aligned[i] and price < ema_50_1d_aligned[i]
        
        if position == 0:
            # Enter long: volume spike + long regime
            if vol_spike and long_regime:
                signals[i] = 0.20
                position = 1
            # Enter short: volume spike + short regime
            elif vol_spike and short_regime:
                signals[i] = -0.20
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below 4h EMA20 OR session ends
            if price < ema_20_4h_aligned[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:
            # Exit short: price breaks above 4h EMA20 OR session ends
            if price > ema_20_4h_aligned[i] or not in_session:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_4h_1d_VolumeSpike_HTFTrendRegime_V1"
timeframe = "1h"
leverage = 1.0