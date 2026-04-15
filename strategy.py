#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and 1d EMA50 trend filter
# Long when price breaks above 1d Camarilla R3 level + volume > 1.8x 20-period avg + price > 1d EMA50
# Short when price breaks below 1d Camarilla S3 level + volume > 1.8x 20-period avg + price < 1d EMA50
# Uses 1d price structure (Camarilla pivots) and 1d EMA for trend alignment on 12h chart
# Designed for low trade frequency (12-25/year) to minimize fee drag while capturing strong trends
# Works in both bull and bear markets by requiring volume confirmation and trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R3, S3) and EMA50 ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pivot_point = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate Camarilla levels
    camarilla_r3 = pivot_point + (high_1d - low_1d) * 1.1 / 4.0
    camarilla_s3 = pivot_point - (high_1d - low_1d) * 1.1 / 4.0
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all 1d indicators to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 60
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.8x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R3 level
        # 2. Volume confirmation
        # 3. Price above 1d EMA50 (uptrend filter)
        if (close[i] > camarilla_r3_aligned[i]) and vol_confirm and (close[i] > ema_50_1d_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S3 level
        # 2. Volume confirmation
        # 3. Price below 1d EMA50 (downtrend filter)
        elif (close[i] < camarilla_s3_aligned[i]) and vol_confirm and (close[i] < ema_50_1d_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_Volume_1dEMA50_Filter_v1"
timeframe = "12h"
leverage = 1.0