#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla Pivot R1/S1 breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above 4h Camarilla R1 + 1d volume > 2x 20-period avg + CHOP > 61.8 (range)
# Short when price breaks below 4h Camarilla S1 + 1d volume > 2x 20-period avg + CHOP > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Target 30-60 trades/year.
# Camarilla pivots provide intraday support/resistance. Volume spike confirms institutional interest.
# Choppiness filter ensures we only trade in ranging markets where mean reversion at pivot levels works.
# Works in bull markets (buy dips to S1 in range) and bear markets (sell rallies to R1 in range).

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
    
    # === 1d Indicator: Volume SMA for spike detection ===
    vol_sma_20_1d = pd.Series(df_1d['volume'].values).rolling(window=20, min_periods=20).mean().values
    vol_sma_20_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20_1d)
    
    # === 4h Indicator: Camarilla Pivot Levels (R1, S1) from previous day ===
    # Camarilla levels: R1 = close + (high - low) * 1.1/12, S1 = close - (high - low) * 1.1/12
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_multiplier = 1.1 / 12
    r1_1d = close_1d + (high_1d - low_1d) * camarilla_multiplier
    s1_1d = close_1d - (high_1d - low_1d) * camarilla_multiplier
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    r1_1d_aligned = align_htf_to_ltf(prices, df_1d, r1_1d)
    s1_1d_aligned = align_htf_to_ltf(prices, df_1d, s1_1d)
    
    # === 4h Indicator: Choppiness Index (CHOP) for regime detection ===
    chop_window = 14
    atr_chop = np.zeros(n)
    for i in range(chop_window, n):
        atr_chop[i] = np.sum(np.maximum(high[i-chop_window+1:i+1] - low[i-chop_window+1:i+1],
                                        np.absolute(high[i-chop_window+1:i+1] - np.roll(close[i-chop_window+1:i+1], 1)),
                                        np.absolute(low[i-chop_window+1:i+1] - np.roll(close[i-chop_window+1:i+1], 1))))
    
    max_high = np.zeros(n)
    min_low = np.zeros(n)
    for i in range(chop_window, n):
        max_high[i] = np.max(high[i-chop_window+1:i+1])
        min_low[i] = np.min(low[i-chop_window+1:i+1])
    
    chop = np.zeros(n)
    for i in range(chop_window, n):
        if max_high[i] != min_low[i]:
            chop[i] = 100 * np.log10(atr_chop[i] / np.log(chop_window) / (max_high[i] - min_low[i])) / np.log10(chop_window)
        else:
            chop[i] = 50  # neutral when no range
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, chop_window) + 5
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current 1d volume > 2x 20-period 1d volume SMA
        vol_spike = df_1d['volume'].values[min(i//24, len(df_1d)-1)] > (vol_sma_20_1d_aligned[i] * 2.0)
        
        # Chop filter: CHOP > 61.8 (ranging market)
        chop_range = chop[i] > 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(r1_1d_aligned[i]) or np.isnan(s1_1d_aligned[i]) or
            np.isnan(vol_sma_20_1d_aligned[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Camarilla R1 (from previous day)
        # 2. Volume spike on 1d timeframe
        # 3. Ranging market (CHOP > 61.8)
        if (close[i] > r1_1d_aligned[i]) and vol_spike and chop_range:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Camarilla S1 (from previous day)
        # 2. Volume spike on 1d timeframe
        # 3. Ranging market (CHOP > 61.8)
        elif (close[i] < s1_1d_aligned[i]) and vol_spike and chop_range:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R1S1_1dVolumeSpike_CHOP_Filter_v1"
timeframe = "4h"
leverage = 1.0