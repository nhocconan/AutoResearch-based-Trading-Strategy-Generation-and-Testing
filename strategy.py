#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d volume confirmation and chop regime filter
# Long when price breaks above 1d Camarilla R3 level + volume > 1.5x 20-period avg + CHOP > 61.8 (range)
# Short when price breaks below 1d Camarilla S3 level + volume > 1.5x 20-period avg + CHOP > 61.8 (range)
# Uses discrete position sizing (0.25) to minimize fee churn. Designed for low trade frequency (12-30/year).
# Camarilla levels provide high-probability reversal/breakout points. CHOP filter ensures we trade in ranging markets where mean reversion works.
# Works in bull markets (buying dips to R3) and bear markets (selling rallies to S3) by fading extremes in ranging regimes.

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
    
    # === 1d Indicator: Camarilla Pivot Levels (R3, S3) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    
    # Camarilla levels
    r3 = pivot + (range_1d * 1.1 / 4)
    s3 = pivot - (range_1d * 1.1 / 4)
    
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # === 12h Indicator: Choppiness Index (CHOP) for regime detection ===
    # CHOP > 61.8 = ranging market (good for mean reversion)
    # CHOP < 38.2 = trending market (avoid for this strategy)
    high_12h = pd.Series(high).rolling(window=12, min_periods=12).max().values  # approximate 12h from 15m? 
    low_12h = pd.Series(low).rolling(window=12, min_periods=12).min().values
    close_12h = pd.Series(close).rolling(window=12, min_periods=12).mean().values
    
    # True Range for 12h approximation
    tr1 = high_12h - low_12h
    tr2 = np.abs(high_12h - np.roll(close_12h, 1))
    tr3 = np.abs(low_12h - np.roll(close_12h, 1))
    tr1[0] = high_12h[0] - low_12h[0]
    tr2[0] = np.abs(high_12h[0] - close_12h[0])
    tr3[0] = np.abs(low_12h[0] - close_12h[0])
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Sum of True Range over 14 periods
    atr_sum = np.zeros_like(tr)
    for i in range(14, len(tr)):
        atr_sum[i] = np.sum(tr[i-13:i+1])  # 14-period sum
    
    # Choppiness Index: CHOP = 100 * log10(ATR_sum / (ATR_period * sqrt(period))) / log10(sqrt(period))
    # Simplified: CHOP = 100 * log10(ATR_sum / (14 * ATR)) / log10(sqrt(14))
    # We'll use a common approximation: CHOP = 100 * log10(ATR_sum / (14 * true_range_avg)) / log10(sqrt(14))
    # For simplicity, we'll use: CHOP = 100 * log10(ATR_sum / (14 * tr)) / log10(np.sqrt(14))
    # But to avoid division by zero and complex calc, we'll use a simpler range-based chop
    # Alternative: CHOP = 100 * log10(sum(tr) / (ATR * N)) / log10(sqrt(N))
    # Let's use a rolling std dev approximation for chop
    
    # Using close prices for chop calculation (more stable)
    close_roll = pd.Series(close).rolling(window=14, min_periods=14)
    atr_approx = close_roll.apply(lambda x: np.sum(np.abs(np.diff(x))) if len(x) > 1 else 0, raw=True)
    price_range = close_roll.max() - close_roll.min()
    chop = np.where(price_range > 0, 100 * np.log10(atr_approx / price_range) / np.log10(np.sqrt(14)), 50)
    chop_values = chop.values
    chop_values[:13] = 50  # fill warmup with neutral
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 14) + 5  # volume(20) + chop(14) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(chop_values[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Chop regime filter: only trade in ranging markets (CHOP > 61.8)
        chop_filter = chop_values[i] > 61.8
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R3 level
        # 2. Volume confirmation
        # 3. Ranging market (CHOP > 61.8)
        if (close[i] > r3_aligned[i]) and \
           vol_confirm and chop_filter:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S3 level
        # 2. Volume confirmation
        # 3. Ranging market (CHOP > 61.8)
        elif (close[i] < s3_aligned[i]) and \
             vol_confirm and chop_filter:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_1dVol_Chop_Filter_v1"
timeframe = "12h"
leverage = 1.0