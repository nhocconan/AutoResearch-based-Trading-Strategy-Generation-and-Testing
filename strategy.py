#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above 1d Camarilla R3 level + volume > 2x 24-period avg + choppiness > 61.8 (range)
# Short when price breaks below 1d Camarilla S3 level + volume > 2x 24-period avg + choppiness > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Camarilla levels provide intraday support/resistance. Volume spike confirms breakout strength.
# Choppiness filter ensures we only trade in ranging markets where mean reversion at extremes works.

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
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 4)
    s3 = pivot - (range_1d * 1.1 / 4)
    
    # Align to lower timeframe (12h)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 12h Indicator: Choppiness Index (14-period) ===
    chop_window = 14
    atr_12h = np.zeros(n)
    tr = np.zeros(n)
    
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
    
    # ATR (Wilder's smoothing)
    atr_12h[chop_window-1] = np.mean(tr[:chop_window])
    for i in range(chop_window, n):
        atr_12h[i] = (atr_12h[i-1] * (chop_window-1) + tr[i]) / chop_window
    
    # Highest high and lowest low over chop_window
    hh = pd.Series(high).rolling(window=chop_window, min_periods=chop_window).max().values
    ll = pd.Series(low).rolling(window=chop_window, min_periods=chop_window).min().values
    
    # Choppiness Index
    chop = np.zeros(n)
    sum_atr = pd.Series(atr_12h).rolling(window=chop_window, min_periods=chop_window).sum().values
    chop = 100 * np.log10(sum_atr / np.log10(chop_window) / (hh - ll))
    chop = np.where((hh - ll) != 0, chop, 50.0)  # avoid division by zero
    
    # Volume SMA for confirmation (using 24-period = 2 days of 12h bars)
    vol_sma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(chop_window, 24) + 5  # Choppiness(14) + volume(24) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 24-period volume SMA
        vol_confirm = volume[i] > (vol_sma_24[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(chop[i]) or np.isnan(vol_sma_24[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R3 level
        # 2. Volume confirmation
        # 3. Choppiness > 61.8 (ranging market)
        if (close[i] > r3_aligned[i]) and \
           vol_confirm and (chop[i] > 61.8):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S3 level
        # 2. Volume confirmation
        # 3. Choppiness > 61.8 (ranging market)
        elif (close[i] < s3_aligned[i]) and \
             vol_confirm and (chop[i] > 61.8):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_1dVolSpike_Chop_Filter_v1"
timeframe = "12h"
leverage = 1.0