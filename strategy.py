#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above 1d Camarilla R1 + 1d volume > 2x 20-period avg + CHOP(14) > 61.8 (range)
# Short when price breaks below 1d Camarilla S1 + 1d volume > 2x 20-period avg + CHOP(14) > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-40 trades/year.
# Camarilla levels provide intraday support/resistance. Volume spike confirms institutional interest.
# Choppiness filter ensures we only trade in ranging markets where mean reversion works.
# Works in bull markets (buying dips to R1) and bear markets (selling rallies to S1) by requiring range regime.

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
    r1 = close_1d + (range_1d * 1.1 / 12)
    s1 = close_1d - (range_1d * 1.1 / 12)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 1d Indicator: Volume Spike (20-period average) ===
    vol_sma_20 = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    
    # === 4h Indicator: Choppiness Index (14-period) ===
    chop_window = 14
    atr_14 = np.zeros(n)
    tr = np.maximum(high - low, np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    
    # Wilder's smoothing for ATR
    atr_14[chop_window-1] = np.mean(tr[:chop_window])
    for i in range(chop_window, n):
        atr_14[i] = (atr_14[i-1] * (chop_window-1) + tr[i]) / chop_window
    
    # Choppiness Index: CHOP = 100 * log10(sum(ATR14)/ (ATR * sqrt(chop_window))) / log10(chop_window)
    sum_atr_14 = np.zeros(n)
    sum_atr_14[chop_window-1] = np.sum(atr_14[:chop_window])
    for i in range(chop_window, n):
        sum_atr_14[i] = sum_atr_14[i-1] - atr_14[i-chop_window] + atr_14[i]
    
    chop = np.zeros(n)
    for i in range(chop_window-1, n):
        if atr_14[i] > 0 and sum_atr_14[i] > 0:
            chop[i] = 100 * np.log10(sum_atr_14[i] / (atr_14[i] * np.sqrt(chop_window))) / np.log10(chop_window)
        else:
            chop[i] = 50.0  # neutral
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, chop_window) + 5  # volume(20) + chop(14) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 2x 20-period volume SMA
        vol_confirm = df_1d['volume'].values[i // 1440] > (vol_sma_20_aligned[i] * 2.0) if i // 1440 < len(df_1d) else False
        
        # Choppiness filter: CHOP > 61.8 (strong ranging market)
        chop_filter = chop[i] > 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_sma_20_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R1
        # 2. Volume confirmation
        # 3. Choppiness regime (range)
        if (close[i] > r1_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S1
        # 2. Volume confirmation
        # 3. Choppiness regime (range)
        elif (close[i] < s1_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R1S1_1dVolumeSpike_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0