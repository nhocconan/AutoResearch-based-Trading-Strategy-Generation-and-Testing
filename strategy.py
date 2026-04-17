#!/usr/bin/env python3
"""
Hypothesis: 1d strategy using 1w Camarilla H3/L3 levels with 1d EMA34 trend filter and volume confirmation
- Uses 1w Camarilla levels from previous completed week for structure
- 1d EMA34 for trend bias (long when price > EMA34, short when price < EMA34)
- Volume confirmation: current volume > 1.5x 20-period MA
- Fixed position size 0.25 to limit fee churn
- ATR-based trailing stop (2.0x ATR) to manage risk
- Designed for lower frequency (target 20-50 trades/year) to work in both bull and bear markets
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for Camarilla levels (HTF)
    df_1w = get_htf_data(prices, '1w')
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate 1w Camarilla levels (H3, L3) from previous completed week
    rng = high_1w - low_1w
    h3 = close_1w + 1.1 * rng / 4
    l3 = close_1w - 1.1 * rng / 4
    # Shift by 1 to use only completed 1w bars (avoid look-ahead)
    h3_prev = np.roll(h3, 1)
    l3_prev = np.roll(l3, 1)
    h3_prev[0] = h3[0]
    l3_prev[0] = l3[0]
    
    # Get 1d data for EMA34 trend filter and ATR (primary timeframe)
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d EMA34
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # ATR (14-period) on 1d for stoploss
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume average (20-period) on 1d
    volume_1d = df_1d['volume'].values
    volume_ma_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 1d timeframe (primary)
    h3_aligned = align_htf_to_ltf(prices, df_1w, h3_prev)
    l3_aligned = align_htf_to_ltf(prices, df_1w, l3_prev)
    ema34_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    volume_ma_aligned = align_htf_to_ltf(prices, df_1d, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_1d, atr_14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 50  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(ema34_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        ema34_val = ema34_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend filter
            # Long: price closes above H3 + volume spike + price > EMA34
            if price > h3_val and vol > 1.5 * vol_ma and price > ema34_val:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.0 * atr_val
            # Short: price closes below L3 + volume spike + price < EMA34
            elif price < l3_val and vol > 1.5 * vol_ma and price < ema34_val:
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

name = "1d_Camarilla_H3L3_1wEMA34_VolumeSpike_ATRTrail"
timeframe = "1d"
leverage = 1.0