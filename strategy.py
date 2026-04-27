#!/usr/bin/env python3
"""
1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_Regime
Hypothesis: Daily Camarilla R1/S1 breakout with 1-week EMA trend filter and volume confirmation.
Only trade breakouts aligned with weekly trend (price > weekly EMA34 for longs, < for shorts) to avoid counter-trend whipsaws.
In ranging markets (weekly ADX < 20), use mean reversion at Camarilla H3/L3 levels.
This adaptive approach captures breakouts in trending weekly markets and mean reversion in ranging weekly markets,
working across both bull and bear regimes by aligning with higher timeframe structure.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1d Camarilla pivot levels (key levels: R1, S1, H3, L3)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    PP = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    H3 = PP + range_1d * 1.1 / 4.0
    L3 = PP - range_1d * 1.1 / 4.0
    R1 = PP + range_1d * 1.0 / 4.0
    S1 = PP - range_1d * 1.0 / 4.0
    
    # Align 1d Camarilla levels to 1d timeframe (no alignment needed as primary is 1d)
    H3_aligned = H3  # Already at 1d frequency
    L3_aligned = L3
    R1_aligned = R1
    S1_aligned = S1
    
    # Weekly trend filter: EMA34 on weekly closes
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Weekly ADX for regime detection (trending if ADX > 25)
    if len(df_1w) < 14:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # True Range
    tr1 = np.abs(high_1w - low_1w)
    tr2 = np.abs(high_1w - np.roll(close_1w, 1))
    tr3 = np.abs(low_1w - np.roll(close_1w, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Directional Movement
    dm_plus = np.where((high_1w - np.roll(high_1w, 1)) > (np.roll(low_1w, 1) - low_1w),
                       np.maximum(high_1w - np.roll(high_1w, 1), 0), 0)
    dm_minus = np.where((np.roll(low_1w, 1) - low_1w) > (high_1w - np.roll(high_1w, 1)),
                        np.maximum(np.roll(low_1w, 1) - low_1w, 0), 0)
    dm_plus[0] = 0
    dm_minus[0] = 0
    
    # Smoothed values
    tr14 = pd.Series(tr).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_plus_14 = pd.Series(dm_plus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    dm_minus_14 = pd.Series(dm_minus).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    
    # DI and DX
    di_plus = 100 * dm_plus_14 / tr14
    di_minus = 100 * dm_minus_14 / tr14
    dx = np.abs(di_plus - di_minus) / (di_plus + di_minus) * 100
    dx = np.where(np.isnan(dx), 0, dx)
    
    # ADX
    adx = pd.Series(dx).ewm(alpha=1/14, adjust=False, min_periods=14).mean().values
    adx_aligned = align_htf_to_ltf(prices, df_1w, adx)
    
    # Volume spike: current volume > 2.0 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_avg)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Warmup: need enough for all indicators
    start_idx = max(100, 34, 14, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(adx_aligned[i]) or
            np.isnan(volume_spike[i])):
            signals[i] = 0.0
            continue
        
        close_val = close[i]
        adx_val = adx_aligned[i]
        ema_trend = ema_34_1w_aligned[i]
        vol_spike = volume_spike[i]
        size = 0.25  # 25% position size
        
        if position == 0:
            # Flat - look for entry based on weekly regime
            if adx_val > 25:  # Weekly trending - breakout strategy
                long_entry = (close_val > R1_aligned[i]) and (close_val > ema_trend) and vol_spike
                short_entry = (close_val < S1_aligned[i]) and (close_val < ema_trend) and vol_spike
                
                if long_entry:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
                elif short_entry:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
            else:  # Weekly ranging (ADX < 25) - mean reversion at H3/L3
                long_entry = (close_val < L3_aligned[i]) and vol_spike
                short_entry = (close_val > H3_aligned[i]) and vol_spike
                
                if long_entry:
                    signals[i] = size
                    position = 1
                    entry_price = close_val
                elif short_entry:
                    signals[i] = -size
                    position = -1
                    entry_price = close_val
        elif position == 1:
            # Long - exit conditions
            if adx_val > 25:  # Weekly trending - exit at S1 retracement or trend reversal
                if close_val < S1_aligned[i] or close_val < ema_trend:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = size
            else:  # Weekly ranging - exit at H3 (profit target) or L3 stop
                if close_val > H3_aligned[i] or close_val < L3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = size
        elif position == -1:
            # Short - exit conditions
            if adx_val > 25:  # Weekly trending - exit at R1 retracement or trend reversal
                if close_val > R1_aligned[i] or close_val > ema_trend:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = -size
            else:  # Weekly ranging - exit at L3 (profit target) or H3 stop
                if close_val < L3_aligned[i] or close_val > H3_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                    entry_price = 0.0
                else:
                    signals[i] = -size
    
    return signals

name = "1d_Camarilla_R1_S1_Breakout_1wTrend_VolumeSpike_Regime"
timeframe = "1d"
leverage = 1.0