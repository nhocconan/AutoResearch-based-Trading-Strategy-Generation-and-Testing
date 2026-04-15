#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above Camarilla R3 (1d) + volume > 2x 20-period avg + CHOP > 61.8 (range)
# Short when price breaks below Camarilla S3 (1d) + volume > 2x 20-period avg + CHOP > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (20-40/year).
# Camarilla levels provide high-probability reversal points in ranging markets.
# CHOP filter ensures we only trade in ranging regimes where mean reversion works best.
# Volume spike confirms institutional participation at pivot levels.

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
    r3 = pivot + (range_1d * 1.1 / 4.0)
    s3 = pivot - (range_1d * 1.1 / 4.0)
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 4h Indicators: Choppiness Index and Volume SMA ===
    # Choppiness Index (14-period)
    chop_window = 14
    atr_14 = np.zeros(n)
    tr = np.zeros(n)
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR calculation
    atr_14[chop_window-1] = np.mean(tr[:chop_window])
    for i in range(chop_window, n):
        atr_14[i] = (atr_14[i-1] * (chop_window-1) + tr[i]) / chop_window
    
    # Choppiness Index
    sum_tr_14 = np.zeros(n)
    max_high_14 = np.zeros(n)
    min_low_14 = np.zeros(n)
    
    sum_tr_14[chop_window-1] = np.sum(tr[:chop_window])
    max_high_14[chop_window-1] = np.max(high[:chop_window])
    min_low_14[chop_window-1] = np.min(low[:chop_window])
    
    for i in range(chop_window, n):
        sum_tr_14[i] = sum_tr_14[i-1] - tr[i-chop_window] + tr[i]
        max_high_14[i] = max(max_high_14[i-1], high[i])
        min_low_14[i] = min(min_low_14[i-1], low[i])
    
    chop = np.zeros(n)
    for i in range(chop_window-1, n):
        if max_high_14[i] != min_low_14[i]:
            chop[i] = 100 * np.log10(sum_tr_14[i] / (max_high_14[i] - min_low_14[i])) / np.log10(chop_window)
        else:
            chop[i] = 50.0  # neutral when no range
    
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
        
        # Choppiness filter: CHOP > 61.8 (strong ranging regime)
        chop_filter = chop[i] > 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (1d)
        # 2. Volume confirmation
        # 3. Ranging regime (CHOP > 61.8)
        if (close[i] > r3_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (1d)
        # 2. Volume confirmation
        # 3. Ranging regime (CHOP > 61.8)
        elif (close[i] < s3_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R3S3_1dVolSpike_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0