#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d volume confirmation and choppiness regime filter
# Long when price breaks above 1d Camarilla R1 + volume > 2x 20-period avg + choppiness < 61.8 (trending)
# Short when price breaks below 1d Camarilla S1 + volume > 2x 20-period avg + choppiness < 61.8 (trending)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Camarilla levels provide high-probability reversal/breakout points. Volume confirms institutional participation.
# Choppiness filter ensures we only trade in trending markets, avoiding whipsaws in ranging conditions.
# Works in bull markets (breakouts continue) and bear markets (breakdowns accelerate) by requiring trending regime.

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
    range_1d = high_1d - low_1d
    r1 = pivot + (range_1d * 1.1 / 12)
    s1 = pivot - (range_1d * 1.1 / 12)
    
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h Indicator: Choppiness Index (trend/range regime filter) ===
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    chop_window = 14
    atr_chop = np.zeros(n)
    tr = np.maximum(high_12h - low_12h, 
                    np.maximum(np.abs(high_12h - np.roll(close_12h, 1)),
                               np.abs(low_12h - np.roll(close_12h, 1))))
    tr[0] = high_12h[0] - low_12h[0]
    
    # ATR calculation for Chop
    atr_chop[chop_window-1] = np.mean(tr[:chop_window])
    for i in range(chop_window, n):
        atr_chop[i] = (atr_chop[i-1] * (chop_window-1) + tr[i]) / chop_window
    
    # Sum of true range over chop_window period
    sum_tr = np.zeros(n)
    sum_tr[chop_window-1] = np.sum(tr[:chop_window])
    for i in range(chop_window, n):
        sum_tr[i] = sum_tr[i-1] - tr[i-chop_window] + tr[i]
    
    # Choppiness Index: 100 * log10(sum(tr)/atr) / log10(chop_window)
    chop = np.zeros(n)
    for i in range(chop_window-1, n):
        if atr_chop[i] > 0:
            chop[i] = 100 * np.log10(sum_tr[i] / atr_chop[i]) / np.log10(chop_window)
        else:
            chop[i] = 50.0  # neutral when ATR is zero
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, chop_window) + 20  # 1d data + Chop(14) + volume(20)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Regime filter: choppiness < 61.8 (trending market)
        trending_regime = chop[i] < 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop[i])):
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

name = "12h_CamarillaR1S1_1dVol2x_CHOP_Filter_v1"
timeframe = "12h"
leverage = 1.0