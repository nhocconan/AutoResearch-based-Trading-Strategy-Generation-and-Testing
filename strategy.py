#!/usr/bin/env python3
"""
4h_TRIX_VolumeSpike_ChopRegime_v1
Hypothesis: Trade 4h TRIX (TRIple Exponential Average) crossovers with volume spike confirmation 
and choppiness regime filter. TRIX > 0 and rising = bullish momentum, TRIX < 0 and falling = bearish.
Only trade when CHOP(14) > 61.8 (ranging market) for mean reversion at extremes or 
CHOP(14) < 38.2 (trending market) for momentum continuation. Volume spike (volume > 2.0 * ATR) 
confirms institutional participation. Target: 20-40 trades/year to minimize fee drag.
Discrete sizing: 0.25 long/short.
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
    
    # Get 12h data for HTF trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # Calculate 12h EMA34 for trend regime
    ema_34_12h = pd.Series(df_12h['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate TRIX on close (primary timeframe: 4h)
    # TRIX = EMA(EMA(EMA(close), period), period), period) - 1
    ema1 = pd.Series(close).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema2 = pd.Series(ema1).ewm(span=15, adjust=False, min_periods=15).mean().values
    ema3 = pd.Series(ema2).ewm(span=15, adjust=False, min_periods=15).mean().values
    trix = (ema3 / np.roll(ema3, 1)) - 1  # percentage change
    trix[0] = 0  # first value undefined
    
    # Calculate ATR for volatility filters
    tr1 = np.maximum(high[1:] - low[1:], np.abs(high[1:] - close[:-1]))
    tr2 = np.maximum(np.abs(low[1:] - close[:-1]), tr1)
    tr = np.concatenate([[np.inf], tr2])
    atr = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    
    # Calculate Choppiness Index (CHOP) for regime detection
    # CHOP = 100 * log10(sum(ATR(14)) / (max(high) - min(low))) / log10(period)
    atr_14 = pd.Series(tr).ewm(span=14, adjust=False, min_periods=14).mean().values
    sum_atr_14 = pd.Series(atr_14).rolling(window=14, min_periods=14).sum().values
    max_high_14 = pd.Series(high).rolling(window=14, min_periods=14).max().values
    min_low_14 = pd.Series(low).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(sum_atr_14 / (max_high_14 - min_low_14)) / np.log10(14)
    # Handle division by zero or invalid values
    chop = np.where((max_high_14 - min_low_14) > 0, chop, 50.0)
    chop = np.where(np.isnan(chop), 50.0, chop)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    bars_since_entry = 0  # track holding period
    
    # Start index: need warmup for TRIX (45), ATR (14), CHOP (14), 12h EMA (34)
    start_idx = max(45, 14, 14, 34)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(trix[i]) or np.isnan(trix[i-1]) or 
            np.isnan(atr[i]) or np.isnan(chop[i]) or
            np.isnan(ema_34_12h_aligned[i])):
            signals[i] = 0.0
            bars_since_entry = 0
            continue
        
        # Volume spike: current volume > 2.0 * ATR
        volume_spike = volume[i] > 2.0 * atr[i]
        
        # TRIX signals: TRIX > 0 and rising = bullish, TRIX < 0 and falling = bearish
        trix_bullish = (trix[i] > 0) and (trix[i] > trix[i-1])
        trix_bearish = (trix[i] < 0) and (trix[i] < trix[i-1])
        
        # Choppiness regime: CHOP > 61.8 = ranging (mean revert), CHOP < 38.2 = trending (momentum)
        chop_ranging = chop[i] > 61.8
        chop_trending = chop[i] < 38.2
        
        # 12h trend filter: only trade in direction of 12h trend
        price_above_12h_ema = close[i] > ema_34_12h_aligned[i]
        price_below_12h_ema = close[i] < ema_34_12h_aligned[i]
        
        if position == 0:
            # Long setup: TRIX bullish AND volume spike AND (chop ranging OR (chop trending AND price above 12h EMA))
            long_setup = trix_bullish and volume_spike and (chop_ranging or (chop_trending and price_above_12h_ema))
            
            # Short setup: TRIX bearish AND volume spike AND (chop ranging OR (chop trending AND price below 12h EMA))
            short_setup = trix_bearish and volume_spike and (chop_ranging or (chop_trending and price_below_12h_ema))
            
            if long_setup:
                signals[i] = 0.25
                position = 1
                bars_since_entry = 0
            elif short_setup:
                signals[i] = -0.25
                position = -1
                bars_since_entry = 0
            else:
                signals[i] = 0.0
                bars_since_entry = 0
        elif position == 1:
            # Long: hold position
            signals[i] = 0.25
            bars_since_entry += 1
            # Exit: TRIX turns bearish OR chop becomes extremely ranging OR max holding period (16 bars = 2 days)
            if (not trix_bullish) or (chop[i] > 70) or (bars_since_entry >= 16):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
        elif position == -1:
            # Short: hold position
            signals[i] = -0.25
            bars_since_entry += 1
            # Exit: TRIX turns bullish OR chop becomes extremely ranging OR max holding period (16 bars = 2 days)
            if (not trix_bearish) or (chop[i] > 70) or (bars_since_entry >= 16):
                signals[i] = 0.0
                position = 0
                bars_since_entry = 0
    
    return signals

name = "4h_TRIX_VolumeSpike_ChopRegime_v1"
timeframe = "4h"
leverage = 1.0