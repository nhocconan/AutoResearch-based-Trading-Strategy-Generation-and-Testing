#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above 1d Camarilla R1 + 12h volume > 2x 20-period avg + 12h chop < 61.8 (trending)
# Short when price breaks below 1d Camarilla S1 + 12h volume > 2x 20-period avg + 12h chop < 61.8
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (15-30/year).
# Camarilla levels provide intraday support/resistance. Volume spike confirms breakout strength.
# Chop filter ensures we trade only in trending markets, avoiding whipsaws in ranging conditions.
# Works in bull markets (buying strength) and bear markets (selling weakness) by requiring trending regime.

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
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h Indicator: Choppiness Index (regime filter) ===
    chop_window = 14
    atr_chop = np.zeros_like(close)
    tr = np.zeros_like(close)
    
    # True Range
    high_shift = np.roll(high, 1)
    low_shift = np.roll(low, 1)
    close_shift = np.roll(close, 1)
    high_shift[0] = high[0]
    low_shift[0] = low[0]
    close_shift[0] = close[0]
    
    tr1 = high - low
    tr2 = np.abs(high - close_shift)
    tr3 = np.abs(low - close_shift)
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation
    atr_chop[chop_window-1] = np.mean(tr[:chop_window])
    for i in range(chop_window, len(tr)):
        atr_chop[i] = (atr_chop[i-1] * (chop_window-1) + tr[i]) / chop_window
    
    # Choppiness Index
    sum_tr = np.zeros_like(close)
    sum_tr[chop_window-1] = np.sum(tr[:chop_window])
    for i in range(chop_window, len(tr)):
        sum_tr[i] = sum_tr[i-1] + tr[i]
    
    highest_high = pd.Series(high).rolling(window=chop_window, min_periods=chop_window).max().values
    lowest_low = pd.Series(low).rolling(window=chop_window, min_periods=chop_window).min().values
    
    chop = np.zeros_like(close)
    for i in range(chop_window-1, len(close)):
        if highest_high[i] != lowest_low[i]:
            chop[i] = 100 * np.log10(sum_tr[i] / (highest_high[i] - lowest_low[i])) / np.log10(chop_window)
        else:
            chop[i] = 50.0  # neutral when no range
    
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)  # Using 1d HTF for chop calculation
    
    # === 12h Indicator: Volume Spike Confirmation ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(chop_window, 20) + 5
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Regime filter: chop < 61.8 (trending market)
        trending_regime = chop_aligned[i] < 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R1
        # 2. Volume confirmation
        # 3. Trending regime (chop < 61.8)
        if (close[i] > r1_aligned[i]) and vol_confirm and trending_regime:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S1
        # 2. Volume confirmation
        # 3. Trending regime (chop < 61.8)
        elif (close[i] < s1_aligned[i]) and vol_confirm and trending_regime:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_1dVolumeSpike_ChopFilter_v1"
timeframe = "12h"
leverage = 1.0