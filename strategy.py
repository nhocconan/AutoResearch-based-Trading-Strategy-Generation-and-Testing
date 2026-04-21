#!/usr/bin/env python3
"""
12h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_V2
Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and chop regime filter.
Long when price breaks above R1 with volume > 1.5x median and CHOP > 61.8 (range).
Short when price breaks below S1 with volume > 1.5x median and CHOP > 61.8.
HTF 1d trend filter: only long when price > EMA50, only short when price < EMA50.
ATR-based stoploss via signal=0 when price moves against position by 2.0*ATR.
Designed for low trade frequency (target: 12-37/year) to minimize fee drag.
Works in ranging markets (chop regime) where Camarilla levels act as support/resistance.
HTF trend filter prevents counter-trend trades in strong trends.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for EMA trend filter and Camarilla calculation)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d OHLC for Camarilla levels (previous day) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for today using yesterday's OHLC
    # R1 = close + (high - low) * 1.1/12
    # S1 = close - (high - low) * 1.1/12
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    
    # First bar will have NaN due to roll, that's fine
    camarilla_range = prev_high - prev_low
    r1 = prev_close + camarilla_range * 1.1 / 12
    s1 = prev_close - camarilla_range * 1.1 / 12
    
    # Align Camarilla levels to 12h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Volume confirmation: volume > 1.5x median volume (using 50-period median)
    volume_median = pd.Series(volume_12h).rolling(window=50, min_periods=50).median().values
    volume_threshold = volume_median * 1.5
    
    # Choppiness Index regime filter (CHOP > 61.8 = ranging market)
    # CHOP = 100 * log10(sum(ATR(1)) / (ATR(14) * sqrt(14))) / log10(14)
    # We'll use a simplified version: CHOP = 100 * log10(TR_sum / (ATR * sqrt(14))) / log10(14)
    tr1 = pd.Series(high_12h - low_12h)
    tr2 = pd.Series(np.abs(high_12h - np.roll(close_12h, 1)))
    tr3 = pd.Series(np.abs(low_12h - np.roll(close_12h, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1 = tr  # ATR(1) is just true range
    atr_14 = tr.rolling(window=14, min_periods=14).mean().values
    
    # Avoid division by zero and log of zero
    tr_sum = tr.rolling(window=14, min_periods=14).sum().values
    chop_raw = 100 * np.log10(tr_sum / (atr_14 * np.sqrt(14))) / np.log10(14)
    # Handle invalid values (set to 50 as neutral)
    chop = np.where((atr_14 > 0) & (tr_sum > 0) & np.isfinite(chop_raw), chop_raw, 50.0)
    
    # ATR (14-period) for stoploss
    atr = atr_14  # already calculated above
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(volume_threshold[i]) 
            or np.isnan(chop[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        volume = volume_12h[i]
        
        if position == 0:
            # Long: price breaks above R1 + volume confirmation + chop regime (ranging) + long bias from HTF
            if (price > r1_aligned[i] and volume > volume_threshold[i] and 
                chop[i] > 61.8 and price > ema_50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
                entry_price = price
            # Short: price breaks below S1 + volume confirmation + chop regime (ranging) + short bias from HTF
            elif (price < s1_aligned[i] and volume > volume_threshold[i] and 
                  chop[i] > 61.8 and price < ema_50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
                entry_price = price
        
        elif position == 1:
            # Check stoploss
            if price < entry_price - 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price breaks below S1 (opposite level) or chop drops (trending)
            elif price < s1_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Check stoploss
            if price > entry_price + 2.0 * atr[i]:
                signals[i] = 0.0
                position = 0
            # Exit conditions: price breaks above R1 (opposite level) or chop drops (trending)
            elif price > r1_aligned[i] or chop[i] < 38.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1_S1_Breakout_Volume_Regime_ATRStop_V2"
timeframe = "12h"
leverage = 1.0