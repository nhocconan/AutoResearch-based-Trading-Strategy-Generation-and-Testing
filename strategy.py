#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime_v1
Hypothesis: TRIX (triple EMA) momentum with volume spike confirmation and choppiness regime filter works in both bull and bear markets.
- Bull: TRIX > 0 + volume spike + chop < 61.8 (trending) → long
- Bear: TRIX < 0 + volume spike + chop < 61.8 (trending) → short
- Chop > 61.8 (ranging) → no new entries, only manage existing positions
- Uses 4h timeframe for optimal trade frequency (target: 20-40 trades/year)
- Volume confirmation reduces false breakouts
- ATR-based trailing stop manages risk
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 1d data for choppiness index calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate TRIX on close prices
    close = prices['close'].values
    # TRIX = EMA(EMA(EMA(close, period), period), period)
    ema1 = pd.Series(close).ewm(span=12, min_periods=12, adjust=False).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, min_periods=12, adjust=False).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, min_periods=12, adjust=False).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = np.nan  # First value is undefined
    
    # Calculate Choppiness Index on 1d data
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr_1d = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) for 1d
    atr_1d = pd.Series(tr_1d).rolling(window=14, min_periods=14).mean().values
    
    # Sum of TR over 14 periods
    sum_tr_14 = pd.Series(tr_1d).rolling(window=14, min_periods=14).sum().values
    
    # Highest high and lowest low over 14 periods
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    
    # Choppiness Index = 100 * log10(sum_tr_14 / (max_high_14 - min_low_14)) / log10(14)
    # Avoid division by zero
    range_14 = max_high_14 - min_low_14
    chop_1d = np.where(range_14 > 0, 100 * np.log10(sum_tr_14 / range_14) / np.log10(14), 100)
    chop_1d = np.where(chop_1d > 100, 100, chop_1d)  # Cap at 100
    chop_1d = np.where(chop_1d < 0, 0, chop_1d)    # Floor at 0
    
    # Align chop to 4h timeframe
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # Volume filter: 20-period average
    vol_ma = prices['volume'].rolling(window=20, min_periods=20).mean().values
    
    # ATR for stoploss (4h)
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = tr2[0] = tr3[0] = np.nan
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    highest_since_entry = 0.0
    lowest_since_entry = 0.0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(trix[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(atr[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        price = close[i]
        volume = prices['volume'].iloc[i]
        
        # Update trailing extremes
        if position == 1:
            highest_since_entry = max(highest_since_entry, price)
        elif position == -1:
            lowest_since_entry = min(lowest_since_entry, price)
        
        # Volume confirmation
        volume_ok = volume > 1.5 * vol_ma[i]
        
        # Choppiness regime: < 61.8 = trending, > 61.8 = ranging
        chop_ok = chop_aligned[i] < 61.8
        
        if position == 0:
            # Long: TRIX > 0 (bullish momentum) + volume + trending regime
            if trix[i] > 0 and volume_ok and chop_ok:
                signals[i] = 0.25
                position = 1
                entry_price = price
                highest_since_entry = price
            # Short: TRIX < 0 (bearish momentum) + volume + trending regime
            elif trix[i] < 0 and volume_ok and chop_ok:
                signals[i] = -0.25
                position = -1
                entry_price = price
                lowest_since_entry = price
        
        elif position == 1:
            # Trailing stop: exit if price drops 2.5 * ATR from highest
            if price < highest_since_entry - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Trailing stop: exit if price rises 2.5 * ATR from lowest
            if price > lowest_since_entry + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0