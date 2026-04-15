#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume spike and chop regime filter
# Long when price breaks above 1d Camarilla R3 + 1d volume > 2x 20-period avg + CHOP(14) > 61.8 (range)
# Short when price breaks below 1d Camarilla S3 + 1d volume > 2x 20-period avg + CHOP(14) > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Camarilla levels provide precise intraday support/resistance. Chop filter ensures we only trade in ranging markets.
# Works in bear markets (2025+) where range-bound behavior dominates, avoiding false breakouts in trends.

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
    
    # Camarilla levels: R3 = close + (high - low) * 1.1/4, S3 = close - (high - low) * 1.1/4
    camarilla_r3 = close_1d + (range_1d * 1.1 / 4)
    camarilla_s3 = close_1d - (range_1d * 1.1 / 4)
    
    # === 1d Indicator: Volume SMA (20-period) ===
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # === 1d Indicator: Choppiness Index (CHOP) ===
    # CHOP = 100 * log10(sum(ATR(14)) / (log10(n) * (max(high) - min(low)))) 
    # Simplified: CHOP = 100 * log10(atr_sum / (log10(14) * (hh - ll))) / log10(14)
    # We'll use: CHOP > 61.8 = ranging market (good for mean reversion)
    atr_period = 14
    tr1 = high_1d - low_1d
    tr2 = np.abs(high_1d - np.roll(close_1d, 1))
    tr3 = np.abs(low_1d - np.roll(close_1d, 1))
    tr1[0] = high_1d[0] - low_1d[0]
    tr2[0] = np.abs(high_1d[0] - close_1d[0])
    tr3[0] = np.abs(low_1d[0] - close_1d[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period-1) + tr[i]) / atr_period
    
    # Calculate CHOP: 100 * log10( sum(ATR(14)) / (log10(14) * (max(high) - min(low)) ) ) / log10(14)
    # Using rolling window of 14 for CHOP
    atr_sum = pd.Series(atr).rolling(window=14, min_periods=14).sum().values
    hh = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    ll = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    chop = 100 * np.log10(atr_sum / (np.log10(14) * (hh - ll))) / np.log10(14)
    # Handle division by zero and invalid values
    chop = np.where((hh - ll) > 0, chop, 50.0)  # Default to neutral when no range
    
    # Align 1d indicators to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    vol_sma_20_aligned = align_htf_to_ltf(prices, df_1d, vol_sma_20)
    chop_aligned = align_htf_to_ltf(prices, df_1d, chop)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14) + 20  # Camarilla needs 20, CHOP needs 14, volume needs 20
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 2x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20_aligned[i] * 2.0)
        
        # Chop filter: CHOP > 61.8 = ranging market (good for mean reversion)
        chop_filter = chop_aligned[i] > 61.8
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_sma_20_aligned[i]) or np.isnan(chop_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R3
        # 2. Volume confirmation (2x avg)
        # 3. Chop filter (ranging market > 61.8)
        if (close[i] > camarilla_r3_aligned[i]) and \
           vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S3
        # 2. Volume confirmation (2x avg)
        # 3. Chop filter (ranging market > 61.8)
        elif (close[i] < camarilla_s3_aligned[i]) and \
             vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_1dVol2x_CHOP_Filter_v4"
timeframe = "12h"
leverage = 1.0