#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout with volume confirmation and 1d EMA trend filter
# Long when price breaks above 1d Camarilla R3 level + volume > 1.5x 20-period avg + price > 1d EMA50
# Short when price breaks below 1d Camarilla S3 level + volume > 1.5x 20-period avg + price < 1d EMA50
# Uses 1d price structure (Camarilla pivots) and 1d EMA for multi-timeframe trend alignment on 12h chart
# Designed for low trade frequency (12-25/year) to minimize fee drag
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Works in both bull and bear markets by requiring volume confirmation and trend alignment

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
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R3, S3) and EMA50 ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point (PP)
    pivot_point_1d = (high_1d + low_1d + close_1d) / 3.0
    
    # Calculate Camarilla levels
    camarilla_r3_1d = pivot_point_1d + (high_1d - low_1d) * 1.1 / 4.0
    camarilla_s3_1d = pivot_point_1d - (high_1d - low_1d) * 1.1 / 4.0
    
    # Calculate 1d EMA50
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 12h timeframe
    camarilla_r3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_1d)
    camarilla_s3_1d_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_1d)
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_1d_aligned[i]) or np.isnan(camarilla_s3_1d_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R3 level
        # 2. Volume confirmation
        # 3. Price above 1d EMA50 (uptrend)
        if (close[i] > camarilla_r3_1d_aligned[i]) and vol_confirm and \
           (close[i] > ema_50_1d_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S3 level
        # 2. Volume confirmation
        # 3. Price below 1d EMA50 (downtrend)
        elif (close[i] < camarilla_s3_1d_aligned[i]) and vol_confirm and \
             (close[i] < ema_50_1d_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_Volume_1dEMA50_Filter_v1"
timeframe = "12h"
leverage = 1.0