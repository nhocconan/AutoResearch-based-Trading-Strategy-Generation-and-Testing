#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R1/S1 breakout with 1d volume spike and chop regime filter
# Long when price breaks above Camarilla R1 (1d) + volume > 2x 20-period avg + CHOP(14) > 61.8 (range)
# Short when price breaks below Camarilla S1 (1d) + volume > 2x 20-period avg + CHOP(14) > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (15-35/year).
# Camarilla pivots provide intraday support/resistance. Volume spike confirms breakout strength.
# Chop regime filter ensures we only trade in ranging markets where mean reversion at pivots works.
# Works in bull markets (buy dips to S1 in range) and bear markets (sell rallies to R1 in range) by requiring chop > 61.8.

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
    R1 = close_1d + (range_1d * 1.1 / 12)
    S1 = close_1d - (range_1d * 1.1 / 12)
    
    # Align to 4h timeframe (wait for 1d bar to close)
    R1_aligned = align_htf_to_ltf(prices, df_1d, R1)
    S1_aligned = align_htf_to_ltf(prices, df_1d, S1)
    
    # === 4h Indicators: Volume Spike and Choppiness Index ===
    # Volume SMA for spike detection (20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Choppiness Index (14-period)
    chop_window = 14
    atr_14 = np.zeros(n)
    for i in range(chop_window, n):
        tr = np.maximum(high[i] - low[i], 
                        np.maximum(np.abs(high[i] - close[i-1]), 
                                   np.abs(low[i] - close[i-1])))
        atr_14[i] = (atr_14[i-1] * (chop_window-1) + tr) / chop_window if i > chop_window else np.mean([np.maximum(high[j] - low[j], 
                                                                                                  np.maximum(np.abs(high[j] - close[j-1]), 
                                                                                                             np.abs(low[j] - close[j-1]))) for j in range(i-chop_window+1, i+1)])
    
    # Initialize first value
    if chop_window < n:
        tr_sum = 0
        for j in range(chop_window):
            tr = np.maximum(high[j] - low[j], 
                            np.maximum(np.abs(high[j] - close[j-1 if j>0 else 0]), 
                                       np.abs(low[j] - close[j-1 if j>0 else 0])))
            tr_sum += tr
        atr_14[chop_window-1] = tr_sum / chop_window
    
    # Calculate highest high and lowest low over chop_window
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(chop_window-1, n):
        highest_high[i] = np.max(high[i-chop_window+1:i+1])
        lowest_low[i] = np.min(low[i-chop_window+1:i+1])
    
    # Chop = 100 * log10(sum(atr14) / (log10(chop_window) * (highest_high - lowest_low)))
    chop = np.full(n, np.nan)
    for i in range(chop_window-1, n):
        if highest_high[i] > lowest_low[i]:
            sum_atr = np.sum(atr_14[i-chop_window+1:i+1])
            chop[i] = 100 * np.log10(sum_atr) / np.log10(chop_window) / np.log10(highest_high[i] - lowest_low[i])
        else:
            chop[i] = 50.0  # neutral when no range
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, chop_window) + 5  # volume(20) + chop(14) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA (strong breakout)
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Chop filter: CHOP(14) > 61.8 (ranging market)
        chop_filter = chop[i] > 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1 (1d)
        # 2. Volume spike confirmation
        # 3. Chop regime filter (range market)
        if (close[i] > R1_aligned[i]) and \
           vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1 (1d)
        # 2. Volume spike confirmation
        # 3. Chop regime filter (range market)
        elif (close[i] < S1_aligned[i]) and \
             vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_CamarillaR1S1_VolumeSpike_Chop_Filter_v1"
timeframe = "4h"
leverage = 1.0