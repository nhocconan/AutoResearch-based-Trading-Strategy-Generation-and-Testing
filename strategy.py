#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 1d volume confirmation and choppiness regime filter
# Long when price breaks above Camarilla R3 (1d) + volume > 2x 20-period avg + CHOP(14) < 38.2 (trending)
# Short when price breaks below Camarilla S3 (1d) + volume > 2x 20-period avg + CHOP(14) < 38.2
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-40 trades/year.
# Camarilla pivots provide mathematically derived support/resistance. Volume confirms institutional interest.
# Choppiness filter ensures we only trade in trending markets, avoiding chop where pivots fail.
# Works in bull markets (breakouts continue) and bear markets (breakdowns accelerate) with ADX-like trend detection via CHOP.

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
    
    # Align to 4h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 4h Indicators: Volume SMA and Choppiness Index ===
    # Volume SMA (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period)
    chop_window = 14
    atr_chop = np.zeros(n)
    for i in range(chop_window, n):
        tr = np.maximum(
            high[i] - low[i],
            np.maximum(
                np.abs(high[i] - close[i-1]),
                np.abs(low[i] - close[i-1])
            )
        )
        atr_chop[i] = tr
    
    # Sum of ATR over window
    sum_atr = pd.Series(atr_chop).rolling(window=chop_window, min_periods=chop_window).sum().values
    
    # Highest high and lowest low over window
    highest_high = pd.Series(high).rolling(window=chop_window, min_periods=chop_window).max().values
    lowest_low = pd.Series(low).rolling(window=chop_window, min_periods=chop_window).min().values
    
    # Chop = 100 * log10(sum(ATR)/log(window) / (HH - LL))
    # Avoid division by zero and log(0)
    hh_ll = highest_high - lowest_low
    chop = np.zeros(n)
    for i in range(chop_window, n):
        if sum_atr[i] > 0 and hh_ll[i] > 0:
            chop[i] = 100 * np.log10(sum_atr[i] / np.log10(chop_window)) / hh_ll[i]
        else:
            chop[i] = 50.0  # neutral
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, chop_window) + 5  # 1d data + volume(20) + chop(14)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Choppiness filter: CHOP < 38.2 = trending market
        chop_filter = chop[i] < 38.2
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3 (1d)
        # 2. Volume confirmation
        # 3. Trending market (CHOP < 38.2)
        if (close[i] > r3_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3 (1d)
        # 2. Volume confirmation
        # 3. Trending market (CHOP < 38.2)
        elif (close[i] < s3_aligned[i]) and vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_CamarillaR3S3_1dVol2x_CHOP_Filter_v2"
timeframe = "4h"
leverage = 1.0