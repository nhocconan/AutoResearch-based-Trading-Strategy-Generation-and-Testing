#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Williams Alligator + 1d Choppiness regime + volume confirmation
# Williams Alligator: Jaw (EMA13, 8 bars offset), Teeth (EMA8, 5 bars offset), Lips (EMA5, 3 bars offset)
# Long when: Lips > Teeth > Jaw (bullish alignment) AND 1d CHOP > 61.8 (ranging) AND volume > 1.3x 20-period MA
# Short when: Lips < Teeth < Jaw (bearish alignment) AND 1d CHOP > 61.8 (ranging) AND volume > 1.3x 20-period MA
# Exit when: Alligator alignment reverses OR CHOP < 38.2 (trending regime) 
# Uses Alligator for trend identification, CHOP for regime filter (favors ranging markets), volume for conviction
# Timeframe: 12h, HTF: 1d for CHOP. Target: 50-150 total trades over 4 years (12-37/year) to avoid fee drag.

name = "12h_WilliamsAlligator_1dCHOP_VolumeConfirm"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Williams Alligator on 12h
    if len(close) >= 13:
        # Jaw: EMA(13) with 8 bars offset
        jaw = pd.Series(close).ewm(span=13, min_periods=13, adjust=False).mean().values
        jaw = np.roll(jaw, 8)  # shift forward 8 bars
        jaw[:8] = np.nan
        
        # Teeth: EMA(8) with 5 bars offset
        teeth = pd.Series(close).ewm(span=8, min_periods=8, adjust=False).mean().values
        teeth = np.roll(teeth, 5)  # shift forward 5 bars
        teeth[:5] = np.nan
        
        # Lips: EMA(5) with 3 bars offset
        lips = pd.Series(close).ewm(span=5, min_periods=5, adjust=False).mean().values
        lips = np.roll(lips, 3)  # shift forward 3 bars
        lips[:3] = np.nan
    else:
        jaw = teeth = lips = np.full(n, np.nan)
    
    # Alligator alignment signals
    bullish_alignment = (lips > teeth) & (teeth > jaw)  # Lips > Teeth > Jaw
    bearish_alignment = (lips < teeth) & (teeth < jaw)  # Lips < Teeth < Jaw
    
    # Volume confirmation on 12h
    if len(volume) >= 20:
        vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        volume_filter = volume > (1.3 * vol_ma_20)
    else:
        volume_filter = np.zeros(n, dtype=bool)
    
    # Get 1d data ONCE before loop for CHOP calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:  # need sufficient data for CHOP
        return np.zeros(n)
    
    # Calculate Choppiness Index (CHOP) on 1d
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    if len(high_1d) >= 14:
        # True Range
        tr1 = np.abs(high_1d[1:] - low_1d[1:])
        tr2 = np.abs(high_1d[1:] - close_1d[:-1])
        tr3 = np.abs(low_1d[1:] - close_1d[:-1])
        tr = np.maximum(np.maximum(tr1, tr2), tr3)
        tr = np.concatenate([[np.nan], tr])  # align with index
        
        # ATR(14) - sum of TR over 14 periods
        atr_14 = np.zeros(len(tr))
        for i in range(len(tr)):
            if i < 14:
                atr_14[i] = np.nan
            elif i == 14:
                atr_14[i] = np.nansum(tr[1:15])  # first ATR
            else:
                atr_14[i] = atr_14[i-1] - (atr_14[i-1]/14) + tr[i]  # Wilder's smoothing
        
        # Highest high and lowest low over 14 periods
        hh_14 = np.zeros(len(high_1d))
        ll_14 = np.zeros(len(low_1d))
        for i in range(len(high_1d)):
            if i < 14:
                hh_14[i] = np.nan
                ll_14[i] = np.nan
            else:
                hh_14[i] = np.nanmax(high_1d[i-13:i+1])
                ll_14[i] = np.nanmin(low_1d[i-13:i+1])
        
        # CHOP = 100 * log10(sum(ATR14) / (HH14 - LL14)) / log10(14)
        chop = np.zeros(len(close_1d))
        for i in range(len(close_1d)):
            if np.isnan(atr_14[i]) or np.isnan(hh_14[i]) or np.isnan(ll_14[i]) or hh_14[i] == ll_14[i]:
                chop[i] = np.nan
            else:
                chop[i] = 100 * np.log10(atr_14[i] / (hh_14[i] - ll_14[i])) / np.log10(14)
    else:
        chop = np.full(len(df_1d), np.nan)
    
    # CHOP regime filter: CHOP > 61.8 = ranging (favor mean reversion), CHOP < 38.2 = trending
    chop_ranging = chop > 61.8
    chop_trending = chop < 38.2
    
    # Align 1d CHOP to 12h timeframe
    chop_ranging_aligned = align_htf_to_ltf(prices, df_1d, chop_ranging.astype(float))
    chop_trending_aligned = align_htf_to_ltf(prices, df_1d, chop_trending.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(bullish_alignment[i]) or np.isnan(bearish_alignment[i]) or 
            np.isnan(volume_filter[i]) or np.isnan(chop_ranging_aligned[i]) or 
            np.isnan(chop_trending_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: bullish alignment + ranging market + volume filter
            if (bullish_alignment[i] and 
                chop_ranging_aligned[i] == 1.0 and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: bearish alignment + ranging market + volume filter
            elif (bearish_alignment[i] and 
                  chop_ranging_aligned[i] == 1.0 and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: alignment reverses OR market becomes trending
            if (not bullish_alignment[i] or chop_trending_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: alignment reverses OR market becomes trending
            if (not bearish_alignment[i] or chop_trending_aligned[i] == 1.0):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals