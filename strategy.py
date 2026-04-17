#!/usr/bin/env python3
"""
Hypothesis: 6h strategy using 12h Supertrend (ATR=10, mult=3.0) for trend direction,
combined with 6h Donchian(20) breakout and volume confirmation.
- Long when: price breaks above Donchian upper (20) + volume > 2.0x 20-period 6h volume MA + 12h Supertrend = uptrend
- Short when: price breaks below Donchian lower (20) + volume > 2.0x 20-period 6h volume MA + 12h Supertrend = downtrend
- Fixed position size 0.25 to limit fee churn and manage drawdown
- ATR-based trailing stop (2.5x ATR) to lock in profits
- Designed for very low trade frequency (target: 50-150 trades over 4 years) to avoid fee drag
- Works in bull markets (buying breakouts with uptrend) and bear markets (selling breakdowns with downtrend)
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
    
    # Calculate ATR(10) on 12h for Supertrend
    tr1_12h = high_12h - low_12h
    tr2_12h = np.abs(high_12h - np.roll(close_12h, 1))
    tr3_12h = np.abs(low_12h - np.roll(close_12h, 1))
    tr_12h = np.maximum(tr1_12h, np.maximum(tr2_12h, tr3_12h))
    tr_12h[0] = tr1_12h[0]
    atr_10_12h = pd.Series(tr_12h).rolling(window=10, min_periods=10).mean().values
    
    # Calculate Supertrend on 12h
    hl2_12h = (high_12h + low_12h) / 2.0
    upper_band_12h = hl2_12h + 3.0 * atr_10_12h
    lower_band_12h = hl2_12h - 3.0 * atr_10_12h
    
    supertrend_12h = np.full_like(close_12h, np.nan, dtype=float)
    direction_12h = np.ones_like(close_12h, dtype=int)  # 1 for uptrend, -1 for downtrend
    
    for i in range(1, len(close_12h)):
        if np.isnan(supertrend_12h[i-1]):
            # Initialize
            if close_12h[i] > upper_band_12h[i-1]:
                supertrend_12h[i] = lower_band_12h[i]
                direction_12h[i] = 1
            else:
                supertrend_12h[i] = upper_band_12h[i]
                direction_12h[i] = -1
        else:
            if direction_12h[i-1] == 1:
                supertrend_12h[i] = max(lower_band_12h[i], supertrend_12h[i-1])
                if close_12h[i] < supertrend_12h[i]:
                    direction_12h[i] = -1
                    supertrend_12h[i] = upper_band_12h[i]
                else:
                    direction_12h[i] = 1
            else:
                supertrend_12h[i] = min(upper_band_12h[i], supertrend_12h[i-1])
                if close_12h[i] > supertrend_12h[i]:
                    direction_12h[i] = 1
                    supertrend_12h[i] = lower_band_12h[i]
                else:
                    direction_12h[i] = -1
    
    # Align Supertrend direction to 6h timeframe
    supertrend_dir_aligned = align_htf_to_ltf(prices, df_12h, direction_12h.astype(float))
    
    # Get 6h data for Donchian breakout and volume confirmation (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    volume_6h = df_6h['volume'].values
    
    # Donchian channels (20-period) on 6h
    donchian_upper_20 = pd.Series(high_6h).rolling(window=20, min_periods=20).max().values
    donchian_lower_20 = pd.Series(low_6h).rolling(window=20, min_periods=20).min().values
    
    # Volume average (20-period) on 6h for confirmation
    volume_ma_20 = pd.Series(volume_6h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (10-period) on 6h for stoploss
    tr1 = high_6h - low_6h
    tr2 = np.abs(high_6h - np.roll(close_6h, 1))
    tr3 = np.abs(low_6h - np.roll(close_6h, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr_10 = pd.Series(tr).rolling(window=10, min_periods=10).mean().values
    
    # Align all indicators to 6h timeframe (primary)
    donchian_upper_aligned = align_htf_to_ltf(prices, df_6h, donchian_upper_20)
    donchian_lower_aligned = align_htf_to_ltf(prices, df_6h, donchian_lower_20)
    volume_ma_aligned = align_htf_to_ltf(prices, df_6h, volume_ma_20)
    atr_aligned = align_htf_to_ltf(prices, df_6h, atr_10)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    entry_price = 0.0
    atr_stop = 0.0
    
    start_idx = 100  # warmup
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_upper_aligned[i]) or np.isnan(donchian_lower_aligned[i]) or 
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i]) or 
            np.isnan(supertrend_dir_aligned[i])):
            signals[i] = 0.0
            continue
        
        donch_up = donchian_upper_aligned[i]
        donch_low = donchian_lower_aligned[i]
        vol_ma = volume_ma_aligned[i]
        atr_val = atr_aligned[i]
        supertrend_dir = supertrend_dir_aligned[i]
        vol = volume[i]
        price = close[i]
        
        if position == 0:
            # Look for breakouts with volume confirmation and 12h Supertrend trend filter
            # Long: price breaks above Donchian upper + volume spike + 12h Supertrend uptrend
            if price > donch_up and vol > 2.0 * vol_ma and supertrend_dir == 1:
                signals[i] = 0.25
                position = 1
                entry_price = price
                atr_stop = entry_price - 2.5 * atr_val
            # Short: price breaks below Donchian lower + volume spike + 12h Supertrend downtrend
            elif price < donch_low and vol > 2.0 * vol_ma and supertrend_dir == -1:
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

name = "6h_Supertrend12h_Donchian20_VolumeSpike_ATRTrail"
timeframe = "6h"
leverage = 1.0