#!/usr/bin/env python3
"""
12h_TRIX_VolumeSpike_ChopRegime_ATRStop
Hypothesis: 12h TRIX (12,20,9) zero-cross with volume confirmation (>1.8x 20-period volume MA) and choppiness regime filter (CHOP(14) between 38.2 and 61.8 for ranging markets). 
ATR trailing stop (2.5x ATR) manages risk. Works in bull via TRIX upcross, in bear via TRIX downcross. 
Position size 0.25 balances risk/return. Target ~12-37 trades/year per symbol (50-150 total over 4 years).
Uses 12h primary timeframe with 1d HTF for trend alignment (EMA50), avoiding overtrading while capturing multi-day moves.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load HTF data ONCE before loop (1d for trend filter)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d EMA50 for trend filter ===
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 12h Indicators (primary timeframe) ===
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # TRIX: triple EMA of ROC, period=(12,20,9)
    roc = pd.Series(close_12h).pct_change(periods=1)
    ema1 = pd.Series(roc).ewm(span=12, adjust=False, min_periods=12).mean()
    ema2 = pd.Series(ema1).ewm(span=20, adjust=False, min_periods=20).mean()
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean()
    trix = ema3.values * 100  # scale for readability
    
    # Volume MA (20-period) for spike detection
    vol_ma = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    
    # ATR (14-period) for stoploss
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.concatenate([[np.nan], np.maximum(tr1, np.maximum(tr2, tr3))])
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index (14-period)
    chop_period = 14
    atr_sum = pd.Series(tr).rolling(window=chop_period, min_periods=chop_period).sum().values
    highest_high = pd.Series(high_12h).rolling(window=chop_period, min_periods=chop_period).max().values
    lowest_low = pd.Series(low_12h).rolling(window=chop_period, min_periods=chop_period).min().values
    # Avoid division by zero
    range_max = highest_high - lowest_low
    range_max = np.where(range_max == 0, 1e-10, range_max)
    chop = 100 * np.log10(atr_sum / range_max) / np.log10(chop_period)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(50, n):
        # Skip if indicators not ready
        if (np.isnan(trix[i]) or np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma[i]) 
            or np.isnan(atr[i]) or np.isnan(chop[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close_12h[i]
        vol = volume_12h[i]
        vol_ok = vol > 1.8 * vol_ma[i]  # volume confirmation (tight to reduce trades)
        chop_ok = (chop[i] >= 38.2) and (chop[i] <= 61.8)  # ranging market regime
        
        if position == 0:
            # Long: TRIX crosses above zero + volume confirmation + chop regime + price > 1d EMA50
            if i > 0 and trix[i-1] <= 0 and trix[i] > 0 and vol_ok and chop_ok and price > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: TRIX crosses below zero + volume confirmation + chop regime + price < 1d EMA50
            elif i > 0 and trix[i-1] >= 0 and trix[i] < 0 and vol_ok and chop_ok and price < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Update highest since entry
            highest_since_entry = max(highest_since_entry, price)
            # ATR trailing stop: exit if price drops 2.5*ATR from highest since entry
            if price < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Update lowest since entry
            lowest_since_entry = min(lowest_since_entry, price)
            # ATR trailing stop: exit if price rises 2.5*ATR from lowest since entry
            if price > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_TRIX_VolumeSpike_ChopRegime_ATRStop"
timeframe = "12h"
leverage = 1.0