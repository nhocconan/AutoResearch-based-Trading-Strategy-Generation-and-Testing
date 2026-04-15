#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with volume confirmation and 4h/1d EMA trend filter
# Long when price breaks above 4h Camarilla R3 level + volume > 1.5x 20-period avg + price > 4h EMA34 + price > 1d EMA50
# Short when price breaks below 4h Camarilla S3 level + volume > 1.5x 20-period avg + price < 4h EMA34 + price < 1d EMA50
# Uses 4h price structure (Camarilla pivots) and 4h/1d EMAs for multi-timeframe trend alignment on 1h chart
# Designed for moderate trade frequency (15-35/year) to balance signal quality and fee drag
# Session filter (08-20 UTC) reduces noise during low-liquidity hours
# Works in both bull and bear markets by requiring volume confirmation and multi-TF trend alignment

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
    
    # Get 4h HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 4h Indicators: Camarilla Pivot Levels (R3, S3) and EMA34 ===
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate pivot point (PP)
    pivot_point_4h = (high_4h + low_4h + close_4h) / 3.0
    
    # Calculate Camarilla levels
    camarilla_r3_4h = pivot_point_4h + (high_4h - low_4h) * 1.1 / 4.0
    camarilla_s3_4h = pivot_point_4h - (high_4h - low_4h) * 1.1 / 4.0
    
    # Calculate 4h EMA34
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 1d Indicator: EMA50 ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 1h timeframe
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
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
        if (np.isnan(camarilla_r3_4h_aligned[i]) or np.isnan(camarilla_s3_4h_aligned[i]) or
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Camarilla R3 level
        # 2. Volume confirmation
        # 3. Price above 4h EMA34 (short-term uptrend)
        # 4. Price above 1d EMA50 (long-term uptrend)
        if (close[i] > camarilla_r3_4h_aligned[i]) and vol_confirm and \
           (close[i] > ema_34_4h_aligned[i]) and (close[i] > ema_50_1d_aligned[i]):
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Camarilla S3 level
        # 2. Volume confirmation
        # 3. Price below 4h EMA34 (short-term downtrend)
        # 4. Price below 1d EMA50 (long-term downtrend)
        elif (close[i] < camarilla_s3_4h_aligned[i]) and vol_confirm and \
             (close[i] < ema_34_4h_aligned[i]) and (close[i] < ema_50_1d_aligned[i]):
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Camarilla_R3S3_Volume_4hEMA34_1dEMA50_Filter_v1"
timeframe = "1h"
leverage = 1.0