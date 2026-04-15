#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above 1d Camarilla R1 + volume > 2x 20-period avg + CHOP > 61.8 (range)
# Short when price breaks below 1d Camarilla S1 + volume > 2x 20-period avg + CHOP > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Camarilla pivots provide precise intraday support/resistance. Volume spike confirms institutional interest.
# Choppiness filter ensures we only trade in ranging markets where mean reversion at pivots works best.
# Works in bull markets (buy dips to S1 in range) and bear markets (sell rallies to R1 in range) by requiring chop regime.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Precompute session hours (08-20 UTC) for filter
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d < 30):
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    # Calculate Camarilla levels
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h Indicator: Choppiness Index (CHOP) ===
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n) * (highest_high - lowest_low))
    atr_period = 14
    chop_window = 14
    
    # Calculate TR
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate CHOP
    sum_atr = np.zeros_like(atr)
    sum_atr[chop_window-1] = np.sum(atr[:chop_window])
    for i in range(chop_window, len(atr)):
        sum_atr[i] = sum_atr[i-1] - atr[i-chop_window] + atr[i]
    
    highest_high = pd.Series(high).rolling(window=chop_window, min_periods=chop_window).max().values
    lowest_low = pd.Series(low).rolling(window=chop_window, min_periods=chop_window).min().values
    
    chop = np.zeros_like(atr)
    for i in range(chop_window-1, len(atr)):
        if sum_atr[i] > 0 and highest_high[i] > lowest_low[i]:
            chop[i] = 100 * np.log10(sum_atr[i] / np.log10(chop_window)) / np.log10(highest_high[i] - lowest_low[i])
        else:
            chop[i] = 50.0  # neutral
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(chop_window, atr_period) + 20  # CHOP(14) + ATR(14) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Chop filter: CHOP > 61.8 (ranging market)
        chop_filter = chop[i] > 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R1
        # 2. Volume confirmation
        # 3. Chop filter (ranging market)
        if (close[i] > r1_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S1
        # 2. Volume confirmation
        # 3. Chop filter (ranging market)
        elif (close[i] < s1_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_1dVolSpike2x_Chop_Filter_v1"
timeframe = "12h"
leverage = 1.0