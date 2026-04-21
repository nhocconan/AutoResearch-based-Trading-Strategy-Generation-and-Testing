#!/usr/bin/env python3
"""
6h_Camarilla_R3_S3_Fade_1dTrend_VolumeConfirm
Hypothesis: Fade at Camarilla R3/S3 levels during 1d trend regime with volume confirmation.
In strong 1d trends (price > EMA50), price often retraces to R3/S3 before continuing.
Long when price <= R3 during 1d uptrend with volume confirmation.
Short when price >= S3 during 1d downtrend with volume confirmation.
Uses discrete sizing (0.25) and ATR-based stop (2.0x) to manage risk.
Target: 60-120 total trades over 4 years (15-30/year) to balance edge and fee drag.
Works in both bull (buy retracements) and bear (sell retracements) markets.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend and Camarilla)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d OHLC for Camarilla pivot calculation ===
    df_1d_open = df_1d['open'].values
    df_1d_high = df_1d['high'].values
    df_1d_low = df_1d['low'].values
    df_1d_close = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    range_1d = df_1d_high - df_1d_low
    r3_1d = df_1d_close + 0.55 * range_1d  # R3 level
    s3_1d = df_1d_close - 0.55 * range_1d  # S3 level
    r4_1d = df_1d_close + 0.825 * range_1d  # R4 level (breakout)
    s4_1d = df_1d_close - 0.825 * range_1d  # S4 level (breakout)
    
    # Align 1d Camarilla levels to 6h timeframe
    r3_1d_aligned = align_htf_to_ltf(prices, df_1d, r3_1d)
    s3_1d_aligned = align_htf_to_ltf(prices, df_1d, s3_1d)
    r4_1d_aligned = align_htf_to_ltf(prices, df_1d, r4_1d)
    s4_1d_aligned = align_htf_to_ltf(prices, df_1d, s4_1d)
    
    # === 1d EMA50 for trend filter ===
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 6h ATR (14-period) for stoploss ===
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # === Volume confirmation (1.3x 20-period MA) ===
    volume = prices['volume'].values
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(r3_1d_aligned[i]) or np.isnan(s3_1d_aligned[i]) 
            or np.isnan(r4_1d_aligned[i]) or np.isnan(s4_1d_aligned[i])
            or np.isnan(ema_50_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        r3 = r3_1d_aligned[i]
        s3 = s3_1d_aligned[i]
        r4 = r4_1d_aligned[i]
        s4 = s4_1d_aligned[i]
        ema_50 = ema_50_1d_aligned[i]
        vol_avg = vol_ma[i]
        
        # Volume regime: current volume > 1.3x average (avoid low-volume fades)
        volume_confirmed = volume_now > 1.3 * vol_avg
        
        if position == 0:
            # Fade logic: long at R3 during uptrend, short at S3 during downtrend
            # Only trade if not breaking through R4/S4 (strong breakout)
            long_condition = (price <= r3) and (price >= s3) and (price > ema_50) and volume_confirmed and (price < r4)
            short_condition = (price >= s3) and (price <= r3) and (price < ema_50) and volume_confirmed and (price > s4)
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss (2.0x ATR)
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (price below EMA50)
            elif price < ema_50:
                signals[i] = 0.0
                position = 0
            # Take profit at opposite S3 level
            elif price >= s3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss (2.0x ATR)
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Trend reversal exit (price above EMA50)
            elif price > ema_50:
                signals[i] = 0.0
                position = 0
            # Take profit at opposite R3 level
            elif price <= r3:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "6h_Camarilla_R3_S3_Fade_1dTrend_VolumeConfirm"
timeframe = "6h"
leverage = 1.0