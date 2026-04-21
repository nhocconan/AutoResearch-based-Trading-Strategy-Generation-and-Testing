#!/usr/bin/env python3
"""
4h_TRIX_9_VolumeSpike_ChopRegime_ATRStop
Hypothesis: 4h TRIX(9) zero-cross with volume confirmation (>1.5x 20-period MA) and choppiness regime filter (CHOP > 61.8 = ranging). 
Long when TRIX crosses above zero, volume confirms, and market is ranging (mean reversion edge in bear markets). 
Short when TRIX crosses below zero, volume confirms, and market is ranging. 
Uses ATR-based stop (2.0x) to manage risk. 
Designed for low trade frequency (~20-40/year) to work in both bull and bear markets via regime adaptation.
In ranging markets (high CHOP), mean reversion at momentum extremes works; in trending markets, we avoid false signals.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # === 1d HTF for choppiness regime (completed 1d bar only) ===
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range for 1d
    tr1 = pd.Series(high_1d - low_1d)
    tr2 = pd.Series(np.abs(high_1d - np.roll(close_1d, 1)))
    tr3 = pd.Series(np.abs(low_1d - np.roll(close_1d, 1)))
    tr_1d = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_1d = tr_1d.rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR(14)) / (max(high) - min(low))) / log10(14)
    sum_atr_14 = pd.Series(atr_1d).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop_1d = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14 + 1e-10)) / np.log10(14)
    chop_1d_aligned = align_htf_to_ltf(prices, df_1d, chop_1d)
    
    # === 4h indicators ===
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # TRIX(9): triple EMA of close, then ROC
    ema1 = pd.Series(close).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema2 = pd.Series(ema1).ewm(span=9, adjust=False, min_periods=9).mean().values
    ema3 = pd.Series(ema2).ewm(span=9, adjust=False, min_periods=9).mean().values
    trix = 100 * (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1)
    trix[0] = np.nan  # first value invalid
    
    # 4h ATR (14-period) for stoploss
    tr1 = pd.Series(high - low)
    tr2 = pd.Series(np.abs(high - np.roll(close, 1)))
    tr3 = pd.Series(np.abs(low - np.roll(close, 1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (1.5x 20-period MA)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    bars_since_entry = 0
    
    for i in range(100, n):
        # Skip if indicators not ready
        if (np.isnan(trix[i]) or np.isnan(atr[i]) or 
            np.isnan(vol_ma[i]) or np.isnan(chop_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
            continue
        
        price = close[i]
        volume_now = volume[i]
        trix_now = trix[i]
        trix_prev = trix[i-1]
        vol_avg = vol_ma[i]
        chop_value = chop_1d_aligned[i]
        
        # Volume confirmation: current volume > 1.5x average
        volume_confirm = volume_now > 1.5 * vol_avg
        
        # Choppiness regime: CHOP > 61.8 = ranging (mean reversion zone)
        is_ranging = chop_value > 61.8
        
        if position == 0:
            # Long: TRIX crosses above zero, volume confirm, ranging market
            long_condition = (trix_prev <= 0) and (trix_now > 0) and volume_confirm and is_ranging
            # Short: TRIX crosses below zero, volume confirm, ranging market
            short_condition = (trix_prev >= 0) and (trix_now < 0) and volume_confirm and is_ranging
            
            if long_condition:
                signals[i] = 0.25
                position = 1
                entry_price = price
                bars_since_entry = 0
            elif short_condition:
                signals[i] = -0.25
                position = -1
                entry_price = price
                bars_since_entry = 0
        
        elif position != 0:
            bars_since_entry += 1
            
            # Minimum holding period of 2 bars to reduce churn
            if bars_since_entry < 2:
                signals[i] = 0.25 if position == 1 else -0.25
                continue
            
            # Check stoploss (2.0x ATR)
            if position == 1:
                if price < entry_price - 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if TRIX crosses below zero (momentum reversal)
                elif trix_now < 0:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if price > entry_price + 2.0 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                # Exit if TRIX crosses above zero (momentum reversal)
                elif trix_now > 0:
                    signals[i] = 0.0
                    position = 0
                    bars_since_entry = 0
                else:
                    signals[i] = -0.25
    
    return signals

name = "4h_TRIX_9_VolumeSpike_ChopRegime_ATRStop"
timeframe = "4h"
leverage = 1.0