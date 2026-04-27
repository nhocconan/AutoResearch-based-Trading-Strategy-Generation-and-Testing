#!/usr/bin/env python3
"""
#100953 - 4h_TRIX_12hVolumeSpike_ChopRegime
Hypothesis: TRIX momentum on 4h with 12h volume spike confirmation and 1d Choppiness regime filter. TRIX crosses above/below zero line with volume surge indicates strong momentum. Choppiness index filters for trending markets only (CHOP < 38.2) to avoid whipsaws in ranging conditions. Works in bull (TRIX up + volume) and bear (TRIX down + volume). Targets 20-40 trades/year to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for volume spike filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    volume_12h = df_12h['volume'].values
    
    # Calculate 12h volume EMA20 for spike detection
    vol_ema_12h = pd.Series(volume_12h).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_ema_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ema_12h)
    
    # Get 1d data for Choppiness index
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate True Range and ADX-like components for Choppiness
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    tr[0] = tr1[0]  # First period
    
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Calculate +DM and -DM
    up_move = high_1d - np.roll(high_1d, 1)
    down_move = np.roll(low_1d, 1) - low_1d
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth DM values
    plus_dm_smooth = pd.Series(plus_dm).rolling(window=14, min_periods=14).sum().values
    minus_dm_smooth = pd.Series(minus_dm).rolling(window=14, min_periods=14).sum().values
    
    # Calculate DI values
    plus_di = 100 * plus_dm_smooth / atr
    minus_di = 100 * minus_dm_smooth / atr
    
    # Calculate DX and ADX
    dx = np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    adx = pd.Series(dx).rolling(window=14, min_periods=14).mean().values
    
    # Choppiness Index: higher = ranging, lower = trending
    # CHOP = 100 * log10(sum(ATR)/ (max(high)-min(low)) ) / log10(period)
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    max_high = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    min_low = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (max_high - min_low)) / np.log10(14)
    chop[np.isnan(chop) | (max_high - min_low) == 0] = 50  # Neutral when range is zero
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    # Calculate TRIX on 4h close
    # TRIX = EMA(EMA(EMA(close, period), period), period) - 1 period ago
    ema1 = pd.Series(close).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema2 = pd.Series(ema1).ewm(span=12, adjust=False, min_periods=12).mean().values
    ema3 = pd.Series(ema2).ewm(span=12, adjust=False, min_periods=12).mean().values
    trix = (ema3 - np.roll(ema3, 1)) / np.roll(ema3, 1) * 100
    trix[0] = 0  # First value
    
    # Volume spike: current 12h volume > 2.0 x EMA20
    volume_spike = volume_12h > (vol_ema_12h * 2.0)
    volume_spike_aligned = align_htf_to_ltf(prices, df_12h, volume_spike)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(trix[i]) or np.isnan(chop_aligned[i]) or 
            np.isnan(volume_spike_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Long condition: TRIX crosses above zero, volume spike, trending market (CHOP < 38.2)
        if (trix[i] > 0 and trix[i-1] <= 0 and 
            volume_spike_aligned[i] and 
            chop_aligned[i] < 38.2):
            signals[i] = 0.25
            position = 1
        # Short condition: TRIX crosses below zero, volume spike, trending market (CHOP < 38.2)
        elif (trix[i] < 0 and trix[i-1] >= 0 and 
              volume_spike_aligned[i] and 
              chop_aligned[i] < 38.2):
            signals[i] = -0.25
            position = -1
        # Exit conditions: TRIX crosses back through zero
        elif position == 1 and trix[i] < 0:
            signals[i] = 0.0
            position = 0
        elif position == -1 and trix[i] > 0:
            signals[i] = 0.0
            position = 0
        # Hold position
        else:
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "4h_TRIX_12hVolumeSpike_ChopRegime"
timeframe = "4h"
leverage = 1.0