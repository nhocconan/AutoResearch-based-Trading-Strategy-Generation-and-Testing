#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with 1d volume spike and choppiness regime filter
# Long when price breaks above 1d Camarilla R1 + 12h volume > 2x 20-period avg + CHOP(14) > 61.8 (range)
# Short when price breaks below 1d Camarilla S1 + 12h volume > 2x 20-period avg + CHOP(14) > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Camarilla levels provide high-probability reversal points in ranging markets. Volume spike confirms breakout validity.
# Chop filter ensures we only trade in ranging conditions where mean reversion works, avoiding strong trends that break Camarilla levels.
# Works in bull markets (buying dips at S1 in range) and bear markets (selling rallies at R1 in range) by requiring choppy regime.

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
    
    # Camarilla levels: R1 = close + (range * 1.1/12), S1 = close - (range * 1.1/12)
    camarilla_r1 = close_1d + (range_1d * 1.1 / 12)
    camarilla_s1 = close_1d - (range_1d * 1.1 / 12)
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 12h Indicator: Volume Spike (20-period avg) ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 12h Indicator: Choppiness Index (CHOP) for regime filter ===
    # CHOP = 100 * log10(sum(ATR(14)) / log10(n)) / log10(n)
    # High CHOP (>61.8) = ranging market (good for mean reversion)
    # Low CHOP (<38.2) = trending market (avoid for Camarilla mean reversion)
    
    # Calculate True Range for CHOP
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = high[0] - low[0]
    tr2[0] = np.abs(high[0] - close[0])
    tr3[0] = np.abs(low[0] - close[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR(14) using Wilder's smoothing
    atr_period = 14
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # CHOP calculation: 100 * log10(sum(ATR(14) over 14 periods) / log10(14)) / log10(14)
    chop_period = 14
    atr_sum = np.zeros_like(atr)
    for i in range(chop_period-1, len(atr)):
        atr_sum[i] = np.sum(atr[i-chop_period+1:i+1])
    
    # Avoid log10(0) or log10(1) issues
    chop = np.zeros_like(atr)
    for i in range(chop_period-1, len(atr)):
        if atr_sum[i] > 0 and i+1 > 1:
            chop[i] = 100 * np.log10(atr_sum[i]) / np.log10(i+1)
        else:
            chop[i] = 50.0  # neutral value
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(30, 20, chop_period) + 5  # 1d data + volume(20) + CHOP(14)
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Chop filter: CHOP > 61.8 (ranging market)
        chop_filter = chop[i] > 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(vol_sma_20[i]) or np.isnan(chop[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R1
        # 2. Volume confirmation
        # 3. Chopping regime (range-bound market)
        if (close[i] > camarilla_r1_aligned[i]) and \
           vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S1
        # 2. Volume confirmation
        # 3. Chopping regime (range-bound market)
        elif (close[i] < camarilla_s1_aligned[i]) and \
             vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_1dVolSpike2x_Chop_Filter_v1"
timeframe = "12h"
leverage = 1.0