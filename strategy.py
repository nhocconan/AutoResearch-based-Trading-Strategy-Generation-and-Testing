#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above 1d Camarilla R1 level + 12h volume > 2x 20-period avg + CHOP(12h) > 61.8 (range)
# Short when price breaks below 1d Camarilla S1 level + 12h volume > 2x 20-period avg + CHOP(12h) > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 50-150 trades over 4 years.
# Camarilla pivots provide precise intraday support/resistance. Volume spike confirms breakout strength.
# Choppiness filter ensures we only trade in ranging markets where mean reversion at pivot levels works.
# Works in bull markets (buying dips to S1 in range) and bear markets (selling rallies to R1 in range).

def generate_signals(prices):
    n = len(prices)
    if n < 50:
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
    pp = (high_1d + low_1d + close_1d) / 3.0
    r1 = pp + (high_1d - low_1d) * 1.1 / 12
    s1 = pp - (high_1d - low_1d) * 1.1 / 12
    
    # Align to 12h timeframe (wait for completed 1d candle)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # === 12h Indicators: Volume Spike + Choppiness Index ===
    # Volume SMA for confirmation (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period) on 12h timeframe
    chop_window = 14
    atr_12h = np.zeros(n)
    for i in range(chop_window, n):
        tr = np.maximum(
            high[i] - low[i],
            np.maximum(
                np.abs(high[i] - close[i-1]),
                np.abs(low[i] - close[i-1])
            )
        )
        atr_12h[i] = tr  # Will smooth below
    
    # True Range smoothing (Wilder's)
    atr_smooth = np.zeros_like(atr_12h)
    atr_smooth[chop_window-1] = np.mean(atr_12h[chop_window-1:2*chop_window-1]) if 2*chop_window-1 < n else np.mean(atr_12h[chop_window-1:])
    for i in range(2*chop_window-1, n):
        atr_smooth[i] = (atr_smooth[i-1] * (chop_window-1) + atr_12h[i]) / chop_window
    
    # Calculate highest high and lowest low over chop_window
    highest_high = np.zeros(n)
    lowest_low = np.zeros(n)
    for i in range(chop_window-1, n):
        highest_high[i] = np.max(high[i-chop_window+1:i+1])
        lowest_low[i] = np.min(low[i-chop_window+1:i+1])
    
    # Choppiness Index formula: CHOP = 100 * log10(sum(ATR)/ (HH - LL)) / log10(chop_window)
    sum_atr = np.zeros(n)
    for i in range(chop_window-1, n):
        sum_atr[i] = np.sum(atr_smooth[i-chop_window+1:i+1])
    
    hh_ll = highest_high - lowest_low
    chop = np.zeros(n)
    mask = (hh_ll > 0) & ~np.isnan(hh_ll)
    chop[mask] = 100 * np.log10(sum_atr[mask] / hh_ll[mask]) / np.log10(chop_window)
    chop = np.where(chop > 100, 100, chop)  # Cap at 100
    chop = np.where(chop < 0, 0, chop)      # Floor at 0
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, 2*chop_window) + 5  # 1d data + volume(20) + CHOP(2*14)
    
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
        # 1. Price breaks above 1d Camarilla R1 level
        # 2. Volume confirmation
        # 3. Choppiness filter (range market)
        if (close[i] > r1_aligned[i]) and vol_confirm and chop_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S1 level
        # 2. Volume confirmation
        # 3. Choppiness filter (range market)
        elif (close[i] < s1_aligned[i]) and vol_confirm and chop_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_CamarillaR1S1_1dVol2x_CHOP_Filter_v1"
timeframe = "12h"
leverage = 1.0