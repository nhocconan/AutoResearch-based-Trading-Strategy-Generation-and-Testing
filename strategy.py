#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above 1d Camarilla R1 + volume > 2x 20-period avg + CHOP > 61.8 (range)
# Short when price breaks below 1d Camarilla S1 + volume > 2x 20-period avg + CHOP > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Camarilla levels provide high-probability reversal points in ranging markets.
# Choppiness filter ensures we only trade in ranging conditions (avoid strong trends where pivots fail).
# Volume spike confirms participation at pivot levels.
# Works in bull markets (buy the dip at S1) and bear markets (sell the rally at R1) by mean-reverting at extremes.

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
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pp = (high_1d + low_1d + close_1d) / 3.0
    # Calculate range
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r1 = pp + (range_1d * 1.1 / 12)
    s1 = pp - (range_1d * 1.1 / 12)
    
    # Align to lower timeframe (12h)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h Indicator: Choppiness Index (CHOP) ===
    # CHOP = 100 * log10(sum(TR over n) / (max(high,n) - min(low,n))) / log10(n)
    # High CHOP (>61.8) = ranging market, Low CHOP (<38.2) = trending market
    chop_window = 14
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of TR over window
    tr_sum = pd.Series(tr).rolling(window=chop_window, min_periods=chop_window).sum().values
    
    # Max high and min low over window
    max_high = pd.Series(high).rolling(window=chop_window, min_periods=chop_window).max().values
    min_low = pd.Series(low).rolling(window=chop_window, min_periods=chop_window).min().values
    
    # Avoid division by zero
    denominator = max_high - min_low
    chop = np.zeros_like(tr_sum)
    mask = denominator != 0
    chop[mask] = 100 * np.log10(tr_sum[mask] / denominator[mask]) / np.log10(chop_window)
    
    # === 12h Indicator: Volume SMA for confirmation ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(chop_window, 20) + 5  # CHOP(14) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Choppiness filter: CHOP > 61.8 (ranging market)
        chop_confirm = chop[i] > 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R1
        # 2. Volume confirmation
        # 3. Chop filter (ranging market)
        if (close[i] > r1_aligned[i]) and vol_confirm and chop_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S1
        # 2. Volume confirmation
        # 3. Chop filter (ranging market)
        elif (close[i] < s1_aligned[i]) and vol_confirm and chop_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_1dVolSpike_Chop_Filter_v2"
timeframe = "12h"
leverage = 1.0