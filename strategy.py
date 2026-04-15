#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla pivot breakout with volume confirmation and 12h EMA trend filter
# Long when price breaks above Camarilla R3 + volume > 1.5x 20-period volume SMA + price > 12h EMA34
# Short when price breaks below Camarilla S3 + volume > 1.5x 20-period volume SMA + price < 12h EMA34
# Uses 12h Camarilla levels for structure and 12h EMA for trend alignment to reduce false breakouts
# Designed for low trade frequency (12-30/year) to minimize fee drag while capturing strong trends
# Works in both bull and bear markets by requiring volume confirmation and trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Camarilla Pivot Levels (based on prior 12h bar) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point
    pivot_12h = (high_12h + low_12h + close_12h) / 3.0
    range_12h = high_12h - low_12h
    
    # Camarilla levels
    r3_12h = pivot_12h + (range_12h * 1.1 / 2)
    s3_12h = pivot_12h - (range_12h * 1.1 / 2)
    
    r3_12h_aligned = align_htf_to_ltf(prices, df_12h, r3_12h)
    s3_12h_aligned = align_htf_to_ltf(prices, df_12h, s3_12h)
    
    # === 12h Indicators: EMA34 for Trend Filter ===
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # Skip if any required data is NaN
        if (np.isnan(r3_12h_aligned[i]) or np.isnan(s3_12h_aligned[i]) or
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 12h Camarilla R3
        # 2. Volume confirmation
        # 3. Price above 12h EMA34 (uptrend filter)
        if (close[i] > r3_12h_aligned[i]) and vol_confirm and (close[i] > ema_34_12h_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 12h Camarilla S3
        # 2. Volume confirmation
        # 3. Price below 12h EMA34 (downtrend filter)
        elif (close[i] < s3_12h_aligned[i]) and vol_confirm and (close[i] < ema_34_12h_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R3S3_Volume_12hEMA34_Filter_v1"
timeframe = "6h"
leverage = 1.0