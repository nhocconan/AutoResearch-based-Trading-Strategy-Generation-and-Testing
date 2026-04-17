#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 12h Supertrend for trend bias, 1d Camarilla H3/L3 breakouts with volume confirmation, and ATR trailing stop.
- Uses 12h Supertrend (ATR=10, mult=3.0) for trend filter (long when Supertrend green, short when red)
- Breakout triggers when price closes beyond 1d H3 (long) or L3 (short) with volume > 2.0x 20-period MA
- Fixed position size 0.25 to limit fee churn and manage drawdown
- ATR-based trailing stop (2.5x ATR) to lock in profits and reduce losses
- Designed to work in bull markets (buying H3 breakouts in uptrends) and bear markets (selling L3 breakdowns in downtrends)
- Uses daily Camarilla levels for stronger, less noisy support/resistance
- Optimized for fewer trades (~150 total over 4 years) to minimize fee drag
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
    
    # Get 12h data for Supertrend trend filter (HTF)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Supertrend on 12h
    atr_period = 10
    multiplier = 3.0
    
    # True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Basic Upperband and Lowerband
    hl2 = (high_12h + low_12h) / 2
    upperband = hl2 + (multiplier * atr)
    lowerband = hl2 - (multiplier * atr)
    
    # Initialize Supertrend
    supertrend = np.full_like(close_12h, np.nan, dtype=float)
    direction = np.full_like(close_12h, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    supertrend[atr_period-1] = upperband[atr_period-1]
    direction[atr_period-1] = 1
    
    for i in range(atr_period, len(close_12h)):
        if close_12h[i] > supertrend[i-1]:
            supertrend[i] = upperband[i]
            direction[i] = 1
        elif close_12h[i] < supertrend[i-1]:
            supertrend[i] = lowerband[i]
            direction[i] = -1
        else:
            supertrend[i] = supertrend[i-1]
            direction[i] = direction[i-1]
            
        # Adjust bands
        if direction[i] == 1:
            if lowerband[i] < lowerband[i-1]:
                lowerband[i] = lowerband[i-1]
            supertrend[i] = lowerband[i]
        else:
            if upperband[i] > upperband[i-1]:
                upperband[i] = upperband[i-1]
            supertrend[i] = upperband[i]
    
    # Get 1d data for Camarilla levels (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate daily Camarilla levels (H3, L3) from previous completed 1d bar
    rng_1d = high_1d - low_1d
    h3_1d = close_1d + 1.1 * rng_1d / 4
    l3_1d = close_1d - 1.1 * rng_1d / 4
    # Shift by 1 to use only completed 1d bars (avoid look-ahead)
    h3_1d_prev = np.roll(h3_1d, 1)
    l3_1d_prev = np.roll(l3_1d, 1)
    h3_1d_prev[0] = h3_1d[0]
    l3_1d_prev[0] = l3_1d[0]
    
    # Get 4h data for volume confirmation and ATR (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Volume average (20-period) on 4h
    volume_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) on 4h for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_14 = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align all indicators to 4h timeframe (primary)
    supertrend_aligned = align_htf_to_ltf(prices, df_12h, direction)  # Use direction directly
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3_1d_prev)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3_1d_prev)
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_14)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(supertrend_aligned[i]) or np.isnan(h3_aligned[i]) or 
            np.isnan(l3_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        st_direction = supertrend_aligned[i]
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend filter
            # Long: price closes above H3 + volume spike + Supertrend uptrend
            if price > h3_val and vol > 2.0 * vol_ma and st_direction == 1:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.5 * atr_val
            # Short: price closes below L3 + volume spike + Supertrend downtrend
            elif price < l3_val and vol > 2.0 * vol_ma and st_direction == -1:
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

name = "4h_Supertrend_12h_Camarilla_H3L3_1d_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0