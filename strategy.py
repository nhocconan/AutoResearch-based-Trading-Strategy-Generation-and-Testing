#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot R1/S1 breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above Camarilla R1 (1d) + volume > 2.0x 20-period avg + CHOP > 61.8 (range)
# Short when price breaks below Camarilla S1 (1d) + volume > 2.0x 20-period avg + CHOP > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (15-35/year).
# Camarilla levels provide high-probability intraday support/resistance. CHOP filter ensures we only trade in ranging markets where mean reversion works.
# Works in ranging markets (2025+ bear/range) by fading false breakouts at extremes.

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
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    r1 = close_1d + (high_1d - low_1d) * 1.1 / 12.0
    s1 = close_1d - (high_1d - low_1d) * 1.1 / 12.0
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
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
    
    # ATR for CHOP denominator
    atr = np.zeros_like(tr)
    atr[chop_window-1] = np.mean(tr[:chop_window])
    for i in range(chop_window, len(tr)):
        atr[i] = (atr[i-1] * (chop_window-1) + tr[i]) / chop_window
    
    # Sum of ATR over window
    atr_sum = np.zeros_like(atr)
    for i in range(chop_window-1, len(atr)):
        if i == chop_window-1:
            atr_sum[i] = np.sum(atr[i-chop_window+1:i+1])
        else:
            atr_sum[i] = atr_sum[i-1] - atr[i-chop_window] + atr[i]
    
    # Maximum true range over window
    max_tr = np.zeros_like(tr)
    for i in range(chop_window-1, len(tr)):
        if i == chop_window-1:
            max_tr[i] = np.max(tr[i-chop_window+1:i+1])
        else:
            max_tr[i] = max(max_tr[i-1], tr[i])
    
    # CHOP = 100 * log10(sum(ATR)/max(TR)) / log10(window)
    chop = np.zeros_like(close)
    for i in range(chop_window-1, len(close)):
        if atr_sum[i] > 0 and max_tr[i] > 0:
            chop[i] = 100 * np.log10(atr_sum[i] / max_tr[i]) / np.log10(chop_window)
        else:
            chop[i] = 50.0  # neutral
    
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
        
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Regime filter: CHOP > 61.8 (strong ranging market)
        chop_filter = chop[i] > 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1 (1d)
        # 2. Volume confirmation
        # 3. Ranging market (CHOP > 61.8)
        if (close[i] > r1_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1 (1d)
        # 2. Volume confirmation
        # 3. Ranging market (CHOP > 61.8)
        elif (close[i] < s1_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R1S1_1dVolumeSpike_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0