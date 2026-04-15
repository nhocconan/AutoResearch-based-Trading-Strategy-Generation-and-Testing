#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R3/S3 breakout with volume confirmation and 1d EMA34 trend filter
# Long when price breaks above Camarilla R3 + volume > 1.4x 20-period avg + price > 1d EMA34
# Short when price breaks below Camarilla S3 + volume > 1.4x 20-period avg + price < 1d EMA34
# Uses 12h Camarilla pivot levels (R3/S3) for stronger breakouts and 1d EMA for trend alignment
# Designed for very low trade frequency (12-25/year) to minimize fee drag while capturing strong trends
# Works in both bull and bear markets by requiring volume confirmation and trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h and 1d HTF data once before loop
    df_12h = get_htf_data(prices, '12h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_12h) < 30 or len(df_1d) < 30:
        return np.zeros(n)
    
    # === 12h Indicators: Camarilla Pivot Levels (based on previous 12h bar) ===
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate pivot point and ranges
    pivot = (high_12h + low_12h + close_12h) / 3.0
    range_hl = high_12h - low_12h
    
    # Camarilla levels: R3 = close + (range * 1.1/4), S3 = close - (range * 1.1/4)
    camarilla_r3 = close_12h + (range_hl * 1.1 / 4)
    camarilla_s3 = close_12h - (range_hl * 1.1 / 4)
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_12h, camarilla_s3)
    
    # === 1d Indicators: EMA34 for Trend Filter ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 50
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.4x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.4)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 12h Camarilla R3
        # 2. Volume confirmation
        # 3. Price above 1d EMA34 (uptrend filter)
        if (close[i] > camarilla_r3_aligned[i]) and vol_confirm and (close[i] > ema_34_1d_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 12h Camarilla S3
        # 2. Volume confirmation
        # 3. Price below 1d EMA34 (downtrend filter)
        elif (close[i] < camarilla_s3_aligned[i]) and vol_confirm and (close[i] < ema_34_1d_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R3S3_Volume_1dEMA34_Filter_v1"
timeframe = "12h"
leverage = 1.0