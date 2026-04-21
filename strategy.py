#!/usr/bin/env python3
"""
12h_HTF_1d_WeeklyPivot_VolumeChop_V1
Hypothesis: 12h Camarilla pivot (R1/S1) breakouts with 1d/1w HTF regime filter (price above/below weekly EMA34) and volume confirmation (>1.5x 20-period volume MA). 
Pivots identify key intraday support/resistance; weekly EMA34 filters for higher-timeframe trend alignment. 
Volume confirmation reduces false breakouts. Chop regime filter (CHOP > 61.8) avoids whipsaw in ranging markets. 
Target 12-37 trades/year (50-150 total over 4 years) on BTC/ETH/SOL.
Uses 12h primary timeframe with 1d HTF for pivot calculation and 1w HTF for EMA trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla pivots, 1w for EMA trend)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 34:
        return np.zeros(n)
    
    # === 1d Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    # Camarilla: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.1 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.1 / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 1w EMA34 for trend filter ===
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (CHOP) regime filter - avoids whipsaw in ranging markets
    # CHOP = 100 * log10(sum(ATR(14)) / log10(highest_high - lowest_low)) / log10(14)
    # Simplified: CHOP > 61.8 = ranging (avoid), CHOP < 38.2 = trending (favor)
    tr_range = np.maximum(high_12h - low_12h, 
                          np.maximum(np.abs(high_12h - np.roll(close_12h, 1)), 
                                     np.abs(low_12h - np.roll(close_12h, 1))))
    # Handle first bar
    tr_range[0] = high_12h[0] - low_12h[0]
    atr14 = pd.Series(tr_range).rolling(window=14, min_periods=14).mean().values
    hh14 = pd.Series(high_12h).rolling(window=14, min_periods=14).max().values
    ll14 = pd.Series(low_12h).rolling(window=14, min_periods=14).min().values
    # Avoid division by zero
    chop_raw = 100 * np.log10(atr14.sum() / np.maximum(hh14 - ll14, 1e-10)) / np.log10(14) if len(atr14) >= 14 else np.full_like(atr14, 50.0)
    # Correct calculation per bar
    chop = np.zeros(n)
    for i in range(14, n):
        atr_sum = pd.Series(tr_range[i-13:i+1]).sum()
        hh = hh14[i]
        ll = ll14[i]
        if hh > ll:
            chop[i] = 100 * np.log10(atr_sum) / np.log10(hh - ll) / np.log10(14)
        else:
            chop[i] = 50.0
    # For i < 14, set neutral
    chop[:14] = 50.0
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) 
            or np.isnan(ema_34_1w_aligned[i]) or np.isnan(vol_ma[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        chop_ok = chop[i] > 61.8  # only trade in ranging markets (mean reversion)
        
        if position == 0:
            # Long: price breaks above camarilla R1 + volume confirmation + weekly uptrend + chop regime
            if price > camarilla_r1_aligned[i] and vol_ok and price > ema_34_1w_aligned[i] and chop_ok:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below camarilla S1 + volume confirmation + weekly downtrend + chop regime
            elif price < camarilla_s1_aligned[i] and vol_ok and price < ema_34_1w_aligned[i] and chop_ok:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price breaks below camarilla S1 or weekly trend changes
            if price < camarilla_s1_aligned[i] or price < ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price breaks above camarilla R1 or weekly trend changes
            if price > camarilla_r1_aligned[i] or price > ema_34_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_HTF_1d_WeeklyPivot_VolumeChop_V1"
timeframe = "12h"
leverage = 1.0