#!/usr/bin/env python3
"""
12h_HTF_1d_Camarilla_R1S1_Breakout_VolumeSpike_ATRFilter_V1
Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation (>1.5x 20-period volume MA) and ATR-based stoploss. Uses 1d HTF for trend filter (price > EMA50 for longs, < EMA50 for shorts). Designed for low trade frequency (target: 50-150 total trades over 4 years) to minimize fee drag and work in both bull/bear markets via trend alignment and volatility filtering.
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
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Camarilla pivot levels (based on previous day's OHLC)
    # R1 = close + 1.1*(high - low)/12
    # S1 = close - 1.1*(high - low)/12
    # We use previous 1d bar to calculate levels for current 12h bar
    prev_close_1d = np.roll(close_1d, 1)
    prev_high_1d = np.roll(df_12h['high'].values, 1)  # Approximate: use 12h high as proxy for simplicity
    prev_low_1d = np.roll(df_12h['low'].values, 1)
    
    # Better: use actual 1d OHLC for Camarilla
    # Since we have df_1d, use its OHLC
    camarilla_r1 = close_1d + 1.1 * (df_1d['high'].values - df_1d['low'].values) / 12
    camarilla_s1 = close_1d - 1.1 * (df_1d['high'].values - df_1d['low'].values) / 12
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) 
            or np.isnan(vol_ma[i]) or np.isnan(atr[i]) or np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        vol_ok = vol > 1.5 * vol_ma[i]  # volume confirmation
        
        if position == 0:
            # Long: price breaks above Camarilla R1 + volume + uptrend filter
            if price > camarilla_r1_aligned[i] and vol_ok and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below Camarilla S1 + volume + downtrend filter
            elif price < camarilla_s1_aligned[i] and vol_ok and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks below Camarilla S1 or loss of volume
            elif price < camarilla_s1_aligned[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit: price breaks above Camarilla R1 or loss of volume
            elif price > camarilla_r1_aligned[i] or not vol_ok:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_HTF_1d_Camarilla_R1S1_Breakout_VolumeSpike_ATRFilter_V1"
timeframe = "12h"
leverage = 1.0