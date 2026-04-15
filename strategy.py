#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation.
# Uses Camarilla pivot levels (R3/S3) from 1d for entry, 1d EMA50 for trend bias, and volume > 1.3x 20-bar SMA for confirmation.
# Designed for low trade frequency (12-37/year) to minimize fee drag. Works in bull/bear: EMA50 avoids counter-trend trades,
# Camarilla levels provide structure in ranging markets and breakouts in trending markets.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R3, S3) and EMA50 ===
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    close_1d = pd.Series(df_1d['close'].values)
    
    # Pivot point
    pivot = (high_1d + low_1d + close_1d) / 3
    # Camarilla levels
    camarilla_r3 = pivot + (high_1d - low_1d) * 1.1 / 4
    camarilla_s3 = pivot - (high_1d - low_1d) * 1.1 / 4
    
    # Align Camarilla levels
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3.values)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3.values)
    
    # 1d EMA50 for trend filter
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R3
        # 2. 1d price above EMA50 (bullish trend bias)
        # 3. Volume confirmation
        if (close[i] > camarilla_r3_aligned[i] and
            close[i] > ema_50_1d_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S3
        # 2. 1d price below EMA50 (bearish trend bias)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s3_aligned[i] and
              close[i] < ema_50_1d_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_EMA50_VolFilter_v1"
timeframe = "12h"
leverage = 1.0