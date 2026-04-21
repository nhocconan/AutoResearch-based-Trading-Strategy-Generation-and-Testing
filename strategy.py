#!/usr/bin/env python3
"""
6h_ElderRay_HTFTrend_VolumeSpike_V1
Hypothesis: 6h Elder Ray (Bull/Bear Power) combined with 1d EMA trend filter and 6h volume spike.
Long when Bull Power > 0 (price > EMA13) AND volume > 1.5x volume MA20 AND price > 1d EMA50.
Short when Bear Power < 0 (price < EMA13) AND volume > 1.5x volume MA20 AND price < 1d EMA50.
ATR-based stoploss via signal=0 when price moves against position by 2.0*ATR.
Target: 12-37 trades/year (50-150 total over 4 years) to minimize fee drag.
Works in both bull/bear markets: Elder Ray shows market strength/weakness, volume spike confirms conviction,
HTF trend filter ensures trading with higher timeframe momentum.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h Indicators (primary timeframe) ===
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 30:
        return np.zeros(n)
    
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Elder Ray components
    ema13 = pd.Series(close_6h).ewm(span=13, adjust=False, min_periods=13).mean().values
    bull_power = high_6h - ema13  # Bull Power: High - EMA13
    bear_power = low_6h - ema13   # Bear Power: Low - EMA13
    
    # Volume spike filter: volume > 1.5x 20-period volume MA
    vol_ma20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume_6h > (1.5 * vol_ma20)
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_6h - low_6h)
    tr2 = pd.Series(np.abs(high_6h - np.roll(close_6h, 1)))
    tr3 = pd.Series(np.abs(low_6h - np.roll(close_6h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(bull_power[i]) or np.isnan(bear_power[i]) or 
            np.isnan(volume_spike[i]) or np.isnan(atr[i]) or 
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_6h[i]
        
        if position == 0:
            # Long: Bull Power > 0 AND volume spike AND long bias from HTF
            if bull_power[i] > 0 and volume_spike[i] and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: Bear Power < 0 AND volume spike AND short bias from HTF
            elif bear_power[i] < 0 and volume_spike[i] and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: Bull Power turns negative OR volume spike ends
            elif bull_power[i] <= 0 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: Bear Power turns positive OR volume spike ends
            elif bear_power[i] >= 0 or not volume_spike[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_ElderRay_HTFTrend_VolumeSpike_V1"
timeframe = "6h"
leverage = 1.0