#!/usr/bin/env python3
"""
Hypothesis: 6h Donchian(20) breakout with weekly pivot regime filter + volume confirmation
- Primary: 6h timeframe for lower frequency (target 12-37 trades/year per symbol)
- HTF: 1d for Donchian channel calculation, 1w for weekly pivot levels (R1/S1)
- Regime: Weekly pivot defines bias - only long above weekly pivot, only short below
- Entry: 6h close breaks Donchian(20) from 1d + volume > 1.5x 20-period MA
- Exit: ATR trailing stop (2.0x ATR) or opposite Donchian break
- Position sizing: Fixed 0.25 (discrete levels to minimize fee churn)
- Works in bull markets (buying breakouts in uptrend) and bear markets (selling breakouts in downtrend)
- Weekly pivot acts as structural regime filter to avoid counter-trend trades
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Donchian channel (using daily candles for structure)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Get 1w data for weekly pivot levels
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate Donchian channel (20-period) on 1d using previous completed bar
    highest_20 = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    lowest_20 = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    # Shift by 1 to use only completed bars (avoid look-ahead)
    upper_band = np.roll(highest_20, 1)
    lower_band = np.roll(lowest_20, 1)
    upper_band[0] = high_1d[0]
    lower_band[0] = low_1d[0]
    
    # Volume average (20-period) on 1d
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) on 1d for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate weekly pivot points (standard formula)
    # Pivot = (H + L + C) / 3
    # R1 = 2*Pivot - L
    # S1 = 2*Pivot - H
    pivot_1w = (high_1w + low_1w + close_1w) / 3.0
    r1_1w = 2.0 * pivot_1w - low_1w
    s1_1w = 2.0 * pivot_1w - high_1w
    
    # Align all indicators to 6h timeframe (primary)
    upper_aligned = align_htf_to_ltf(prices, df_1d, upper_band)
    lower_aligned = align_htf_to_ltf(prices, df_1d, lower_band)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    r1_aligned = align_htf_to_ltf(prices, df_1w, r1_1w)
    s1_aligned = align_htf_to_ltf(prices, df_1w, s1_1w)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(upper_aligned[i]) or np.isnan(lower_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        upper = upper_aligned[i]
        lower = lower_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        r1 = r1_aligned[i]
        s1 = s1_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and regime filter
            # Long: price breaks above upper band + volume spike + price > weekly R1 (bullish regime)
            if price > upper and vol > 1.5 * vol_ma and price > r1:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price breaks below lower band + volume spike + price < weekly S1 (bearish regime)
            elif price < lower and vol > 1.5 * vol_ma and price < s1:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.0 * atr_val
        
        elif position == 1:
            # Check stoploss
            if price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 1.5 * atr_val)
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 1.5 * atr_val)
    
    return signals

name = "6h_Donchian20_WeeklyPivotRegime_VolumeSpike_ATRTrail"
timeframe = "6h"
leverage = 1.0