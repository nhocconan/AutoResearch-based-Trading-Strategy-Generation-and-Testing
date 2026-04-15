#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above Camarilla R3 (1d) + volume > 2x 20-period avg + CHOP > 61.8 (range)
# Short when price breaks below Camarilla S3 (1d) + volume > 2x 20-period avg + CHOP > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (20-40/year).
# Camarilla pivots provide mathematical support/resistance levels. Choppiness filter ensures we trade in ranging markets where mean reversion works.
# Works in bull markets (buying dips to support) and bear markets (selling rallies to resistance) by fading extremes in choppy conditions.

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
    
    # === 1d Indicator: Camarilla Pivot Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels
    pivot = (high_1d + low_1d + close_1d) / 3
    range_1d = high_1d - low_1d
    
    # Resistance levels
    r3 = pivot + (range_1d * 1.1 / 4)
    # Support levels
    s3 = pivot - (range_1d * 1.1 / 4)
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 4h Indicator: Choppiness Index (CHOP) ===
    chop_window = 14
    atr_chop = np.zeros_like(close)
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Wilder's smoothing for ATR
    atr_chop[chop_window-1] = np.mean(tr[:chop_window])
    for i in range(chop_window, len(tr)):
        atr_chop[i] = (atr_chop[i-1] * (chop_window-1) + tr[i]) / chop_window
    
    # Calculate highest high and lowest low over chop_window
    hh = pd.Series(high).rolling(window=chop_window, min_periods=chop_window).max().values
    ll = pd.Series(low).rolling(window=chop_window, min_periods=chop_window).min().values
    
    # Avoid division by zero
    sum_atr = atr_chop * chop_window
    chop = np.where((hh - ll) != 0, 100 * np.log10(sum_atr / (hh - ll)) / np.log10(chop_window), 50)
    
    # Volume SMA for confirmation (using 20-period)
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
        
        # Chop filter: CHOP > 61.8 (ranging market)
        chop_filter = chop[i] > 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (1d)
        # 2. Volume confirmation
        # 3. Chop filter (ranging market)
        if (close[i] > r3_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (1d)
        # 2. Volume confirmation
        # 3. Chop filter (ranging market)
        elif (close[i] < s3_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_CamarillaR3S3_1dVolume2x_CHOP_Filter_v1"
timeframe = "4h"
leverage = 1.0