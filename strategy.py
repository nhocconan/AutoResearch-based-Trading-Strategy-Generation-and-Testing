#!/usr/bin/env python3
"""
6h_trix_volume_regime_v1
Hypothesis: TRIX (triple exponential average) with volume confirmation and regime filter (Choppiness Index) works on 6h timeframe.
TRIX > 0 and rising indicates bullish momentum; TRIX < 0 and falling indicates bearish momentum.
Volume confirms momentum strength. Choppiness Index > 61.8 indicates ranging market (use mean reversion at TRIX extremes),
while < 38.2 indicates trending market (follow TRIX direction). Works in both bull and bear markets by adapting to regime.
Targets 12-37 trades/year (50-150 over 4 years).
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_trix_volume_regime_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 6h data for TRIX calculation
    # TRIX = EMA(EMA(EMA(close, period), period), period) - 1-period percent change
    period = 12
    ema1 = pd.Series(close).ewm(span=period, adjust=False).mean()
    ema2 = ema1.ewm(span=period, adjust=False).mean()
    ema3 = ema2.ewm(span=period, adjust=False).mean()
    trix = ema3.pct_change(1) * 100  # percentage change
    trix_values = trix.values
    
    # 12h data for Choppiness Index (regime filter)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate True Range and ATR(14) for Choppiness
    tr1 = high_12h[1:] - low_12h[1:]
    tr2 = np.abs(high_12h[1:] - close_12h[:-1])
    tr3 = np.abs(low_12h[1:] - close_12h[:-1])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr = np.concatenate([[np.nan], tr])  # align with index
    
    atr_period = 14
    atr = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).mean().values
    
    # Highest high and lowest low over ATR period
    hh = pd.Series(high_12h).rolling(window=atr_period, min_periods=atr_period).max().values
    ll = pd.Series(low_12h).rolling(window=atr_period, min_periods=atr_period).min().values
    
    # Choppiness Index: 100 * log10(sum(TR, atr_period) / (hh - ll)) / log10(atr_period)
    tr_sum = pd.Series(tr).rolling(window=atr_period, min_periods=atr_period).sum().values
    chop = 100 * np.log10(tr_sum / (hh - ll)) / np.log10(atr_period)
    
    # Align 12h data to 6h timeframe
    chop_6h = align_htf_to_ltf(prices, df_12h, chop)
    
    # Volume confirmation: 20-period average
    vol_sma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(trix_values[i]) or np.isnan(chop_6h[i]) or 
            np.isnan(vol_sma[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.3x average volume
        vol_confirm = volume[i] > 1.3 * vol_sma[i]
        
        if position == 1:  # Long position
            # Exit: TRIX crosses below zero OR chop > 61.8 and TRIX < 0 (range reversal)
            if trix_values[i] < 0 or (chop_6h[i] > 61.8 and trix_values[i] < 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
        elif position == -1:  # Short position
            # Exit: TRIX crosses above zero OR chop > 61.8 and TRIX > 0 (range reversal)
            if trix_values[i] > 0 or (chop_6h[i] > 61.8 and trix_values[i] > 0):
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Trending market (chop < 38.2): follow TRIX direction
            if chop_6h[i] < 38.2 and vol_confirm:
                if trix_values[i] > 0 and trix_values[i] > trix_values[i-1]:
                    position = 1
                    signals[i] = 0.25
                elif trix_values[i] < 0 and trix_values[i] < trix_values[i-1]:
                    position = -1
                    signals[i] = -0.25
            # Ranging market (chop > 61.8): mean reversion at TRIX extremes
            elif chop_6h[i] > 61.8 and vol_confirm:
                if trix_values[i] < -2.0:  # oversold
                    position = 1
                    signals[i] = 0.25
                elif trix_values[i] > 2.0:  # overbought
                    position = -1
                    signals[i] = -0.25
    
    return signals