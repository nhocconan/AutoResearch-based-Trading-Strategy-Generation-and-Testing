#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 1d Camarilla pivot levels (H3/L3) with 12h EMA34 trend filter and volume confirmation.
Long when price touches/breaks above 1d H3 pivot + volume > 1.8x 20-period 4h volume MA + price above 12h EMA34.
Short when price touches/breaks below 1d L3 pivot + volume > 1.8x 20-period 4h volume MA + price below 12h EMA34.
Fixed position size 0.25 to limit fee churn. ATR(14) trailing stop (2.5x ATR) for risk management.
Designed for low trade frequency (target: 75-200 total over 4 years) to avoid fee drag.
Works in bull markets (buying H3 breakouts above 12h EMA34) and bear markets (selling L3 breakdowns below 12h EMA34).
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
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla pivot levels (H3, L3)
    # Pivot = (H + L + C) / 3
    # Range = H - L
    # H3 = Pivot + (Range * 1.1 / 2)
    # L3 = Pivot - (Range * 1.1 / 2)
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    h3_1d = pivot_1d + (range_1d * 1.1 / 2.0)
    l3_1d = pivot_1d - (range_1d * 1.1 / 2.0)
    
    # Get 12h EMA34 for trend filter
    ema_34_12h = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_1d, ema_34_12h)
    
    # Align Camarilla levels to 4h timeframe
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Get 4h data for volume confirmation and ATR (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Volume average (20-period) on 4h for confirmation
    volume_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) on 4h for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to 4h timeframe (primary)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        ema_34_val = ema_34_12h_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and 12h EMA34 trend filter
            # Long: price touches/breaks above H3 + volume spike + price above 12h EMA34
            if price >= h3_val and vol > 1.8 * vol_ma and price > ema_34_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.5 * atr_val
            # Short: price touches/breaks below L3 + volume spike + price below 12h EMA34
            elif price <= l3_val and vol > 1.8 * vol_ma and price < ema_34_val:
                signals[i] = -0.25
                position = -1
                entry_price = price
                atr_stop = entry_price + 2.5 * atr_val
        
        elif position == 1:
            # Check stoploss
            if price <= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                # Trail stop: raise stop if price moves favorably
                atr_stop = max(atr_stop, price - 2.0 * atr_val)
        
        elif position == -1:
            # Check stoploss
            if price >= atr_stop:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
                # Trail stop: lower stop if price moves favorably
                atr_stop = min(atr_stop, price + 2.0 * atr_val)
    
    return signals

name = "4h_Camarilla_H3L3_12hEMA34_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0