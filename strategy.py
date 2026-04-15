#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume spike filter and choppiness regime filter
# Long when price breaks above 1d Camarilla R1 level + volume > 2x 20-period average + CHOP(14) < 61.8 (trending regime)
# Short when price breaks below 1d Camarilla S1 level + volume > 2x 20-period average + CHOP(14) < 61.8 (trending regime)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Camarilla pivots provide intraday support/resistance levels that work well in ranging and trending markets.
# Volume spike confirms institutional interest. Chop filter ensures we only trade in trending markets, avoiding chop.

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
    
    # Calculate Camarilla levels
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12)
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 12h Indicator: Choppiness Index (CHOP) for regime filter ===
    high_12h = prices['high'].values
    low_12h = prices['low'].values
    close_12h = prices['close'].values
    
    # Calculate True Range
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = high_12h[0] - low_12h[0]
    tr2[0] = np.abs(high_12h[0] - close_12h[0])
    tr3[0] = np.abs(low_12h[0] - close_12h[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR (14-period)
    atr_period = 14
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate sum of True Range over atr_period
    tr_sum = np.zeros_like(tr)
    for i in range(atr_period-1, len(tr)):
        if i == atr_period-1:
            tr_sum[i] = np.sum(tr[i-atr_period+1:i+1])
        else:
            tr_sum[i] = tr_sum[i-1] - tr[i-atr_period] + tr[i]
    
    # Calculate Choppiness Index: CHOP = 100 * log10(tr_sum / (atr * atr_period)) / log10(atr_period)
    chop = np.zeros_like(tr)
    for i in range(atr_period-1, len(tr)):
        if atr[i] > 0 and tr_sum[i] > 0:
            chop[i] = 100 * np.log10(tr_sum[i] / (atr[i] * atr_period)) / np.log10(atr_period)
        else:
            chop[i] = 50.0  # neutral value when calculation invalid
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(atr_period, 20) + 5  # CHOP(14) + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Regime filter: CHOP < 61.8 (trending market)
        trending_regime = chop[i] < 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R1 level
        # 2. Volume confirmation
        # 3. Trending regime (CHOP < 61.8)
        if (close[i] > camarilla_r1_aligned[i]) and \
           vol_confirm and trending_regime:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S1 level
        # 2. Volume confirmation
        # 3. Trending regime (CHOP < 61.8)
        elif (close[i] < camarilla_s1_aligned[i]) and \
             vol_confirm and trending_regime:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_1dVolSpike_Chop_Filter_v1"
timeframe = "12h"
leverage = 1.0