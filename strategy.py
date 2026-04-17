#!/usr/bin/env python3
"""
Hypothesis: 4h strategy using 12h Supertrend for trend filter, 1d Camarilla H3/L3 breakout with volume confirmation
- Uses 12h Supertrend (ATR=10, mult=3.0) for trend bias (long when Supertrend up, short when down)
- Breakout triggers when price closes beyond 1d H3 (long) or L3 (short) with volume > 2.0x 20-period 4h MA
- Fixed position size 0.25 to limit fee churn and manage drawdown
- ATR-based trailing stop (2.5x ATR) to lock in profits and reduce losses
- Combines strong trend filter (12h Supertrend) with precise entry (1d Camarilla breakout) + volume confirmation
- Designed for low trade frequency (target: 75-150 trades over 4 years) to avoid fee drag
- Works in bull markets (buying H3 breakouts in uptrends) and bear markets (selling L3 breakdowns in downtrends)
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
    
    # Get 1d data for Camarilla pivots (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Camarilla levels (based on previous day's range)
    # H3 = close + 1.1*(high-low)/2, L3 = close - 1.1*(high-low)/2
    # Using previous day's high/low to avoid look-ahead
    prev_high_1d = np.roll(high_1d, 1)
    prev_low_1d = np.roll(low_1d, 1)
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d[0] = high_1d[0]  # first period
    prev_low_1d[0] = low_1d[0]
    prev_close_1d[0] = close_1d[0]
    
    camarilla_h3 = prev_close_1d + 1.1 * (prev_high_1d - prev_low_1d) / 2
    camarilla_l3 = prev_close_1d - 1.1 * (prev_high_1d - prev_low_1d) / 2
    
    # Get 12h data for Supertrend (HTF)
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate ATR for Supertrend (10-period)
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]  # first period
    atr_10_12h = pd.Series(tr_12h).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Supertrend
    hl2_12h = (high_12h + low_12h) / 2
    upper_band_12h = hl2_12h + 3.0 * atr_10_12h
    lower_band_12h = hl2_12h - 3.0 * atr_10_12h
    
    supertrend_12h = np.full_like(close_12h, np.nan, dtype=float)
    direction_12h = np.full_like(close_12h, 1, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if close_12h[i-1] > upper_band_12h[i-1]:
            direction_12h[i] = -1
        elif close_12h[i-1] < lower_band_12h[i-1]:
            direction_12h[i] = 1
        else:
            direction_12h[i] = direction_12h[i-1]
        
        if direction_12h[i] == 1:
            supertrend_12h[i] = max(lower_band_12h[i], supertrend_12h[i-1])
        else:
            supertrend_12h[i] = min(upper_band_12h[i], supertrend_12h[i-1])
    
    # Get 4h data for volume confirmation and ATR (primary timeframe)
    df_4h = get_htf_data(prices, '4h')
    volume_4h = df_4h['volume'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Volume average (20-period) on 4h
    volume_ma_20 = pd.Series(volume_4h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (10-period) on 4h for stoploss
    tr1 = high_4h - low_4h
    tr2 = np.abs(high_4h - np.roll(close_4h, 1))
    tr3 = np.abs(low_4h - np.roll(close_4h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # first period
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align all indicators to 4h timeframe (primary)
    h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, direction_12h.astype(float))
    volume_ma_aligned = align_htf_to_ltf(prices, df_4h, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_4h, atr_10)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or 
            np.isnan(supertrend_dir_aligned[i]) or np.isnan(volume_ma_aligned[i]) or 
            np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        h3_val = h3_aligned[i]
        l3_val = l3_aligned[i]
        supertrend_dir = supertrend_dir_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and trend filter
            # Long: price closes above H3 + volume spike + Supertrend up (1)
            if price > h3_val and vol > 2.0 * vol_ma and supertrend_dir == 1:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.5 * atr_val
            # Short: price closes below L3 + volume spike + Supertrend down (-1)
            elif price < l3_val and vol > 2.0 * vol_ma and supertrend_dir == -1:
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

name = "4h_Supertrend12h_Camarilla_H3L3_VolumeSpike_ATRTrail"
timeframe = "4h"
leverage = 1.0