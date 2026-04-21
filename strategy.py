#!/usr/bin/env python3
"""
4h_Camarilla_R3_S3_Breakout_Volume_SuperTrend_v2
Hypothesis: Price breaking above Camarilla R3 or below S3 from prior 1d session captures strong breakouts. Volume confirmation (>1.8x 20-period MA) filters weak moves. SuperTrend(10,3.0) on 4h acts as trend filter: only take longs when SuperTrend is bullish, shorts when bearish. This reduces false breakouts in choppy/range markets. Designed for low trade frequency (~15-30/year) to minimize fee drag and work in both bull and bear regimes by requiring strong momentum + trend alignment.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for Camarilla levels)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # === Camarilla levels from prior 1-day session (HLC of previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R3, S3 levels (stronger breakout signals)
    camarilla_r3 = close_1d + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = close_1d - (high_1d - low_1d) * 1.1 / 4
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # === Volume spike filter (20-period) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / vol_ma
    
    # === SuperTrend on 4h (10, 3.0) ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]
    atr = pd.Series(tr).ewm(span=10, adjust=False, min_periods=10).mean().values
    
    # SuperTrend calculation
    hl2 = (high + low) / 2
    upperband = hl2 + 3.0 * atr
    lowerband = hl2 - 3.0 * atr
    
    supertrend = np.full(n, np.nan)
    direction = np.full(n, 1)  # 1 for uptrend, -1 for downtrend
    
    supertrend[0] = hl2[0]
    direction[0] = 1
    
    for i in range(1, n):
        if close[i] > supertrend[i-1]:
            direction[i] = 1
        else:
            direction[i] = -1
        
        if direction[i] == 1:
            supertrend[i] = max(lowerband[i], supertrend[i-1])
        else:
            supertrend[i] = min(upperband[i], supertrend[i-1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(vol_ratio[i]) or np.isnan(supertrend[i]) or
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price_close = prices['close'].iloc[i]
        price_high = prices['high'].iloc[i]
        price_low = prices['low'].iloc[i]
        vol_spike = vol_ratio[i]
        st_dir = direction[i]
        r3 = camarilla_r3_aligned[i]
        s3 = camarilla_s3_aligned[i]
        
        if position == 0:
            # Long: price breaks above R3 + volume spike > 1.8 + SuperTrend bullish
            if price_close > r3 and vol_spike > 1.8 and st_dir == 1:
                signals[i] = 0.25
                position = 1
                entry_price = price_close
            # Short: price breaks below S3 + volume spike > 1.8 + SuperTrend bearish
            elif price_close < s3 and vol_spike > 1.8 and st_dir == -1:
                signals[i] = -0.25
                position = -1
                entry_price = price_close
        
        elif position != 0:
            # Stoploss: 2.5 * ATR from entry (wider stop for less whipsaw)
            if position == 1:
                if price_close < entry_price - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price_close > entry_price + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_Camarilla_R3_S3_Breakout_Volume_SuperTrend_v2"
timeframe = "4h"
leverage = 1.0