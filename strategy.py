#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R1/S1 breakout with volume confirmation and 1d/1w EMA trend filter
# Long when price breaks above 6h Camarilla R1 level + volume > 1.3x 20-period avg + price > 1d EMA34 + price > 1w EMA50
# Short when price breaks below 6h Camarilla S1 level + volume > 1.3x 20-period avg + price < 1d EMA34 + price < 1w EMA50
# Uses 6h price structure (Camarilla pivots) and 1d/1w EMAs for multi-timeframe trend alignment
# Designed for low trade frequency (12-30/year) to minimize fee drag while capturing institutional breakouts
# Works in both bull and bear markets by requiring volume confirmation and multi-TF trend alignment from higher timeframes

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
    
    # Get 6h HTF data once before loop (primary timeframe)
    df_6h = get_htf_data(prices, '6h')
    if len(df_6h) < 50:
        return np.zeros(n)
    
    # Get 1d HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 30:
        return np.zeros(n)
    
    # === 6h Indicators: Camarilla Pivot Levels (R1, S1) ===
    high_6h = df_6h['high'].values
    low_6h = df_6h['low'].values
    close_6h = df_6h['close'].values
    
    # Calculate pivot point (PP)
    pivot_point_6h = (high_6h + low_6h + close_6h) / 3.0
    
    # Calculate Camarilla levels (R1, S1)
    camarilla_r1_6h = pivot_point_6h + (high_6h - low_6h) * 1.1 / 12.0
    camarilla_s1_6h = pivot_point_6h - (high_6h - low_6h) * 1.1 / 12.0
    
    # === 1d Indicator: EMA34 ===
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # === 1w Indicator: EMA50 ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 6h timeframe
    camarilla_r1_6h_aligned = align_htf_to_ltf(prices, df_6h, camarilla_r1_6h)
    camarilla_s1_6h_aligned = align_htf_to_ltf(prices, df_6h, camarilla_s1_6h)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_6h_aligned[i]) or np.isnan(camarilla_s1_6h_aligned[i]) or
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(ema_50_1w_aligned[i]) or
            np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 6h Camarilla R1 level
        # 2. Volume confirmation
        # 3. Price above 1d EMA34 (short-term uptrend)
        # 4. Price above 1w EMA50 (long-term uptrend)
        if (close[i] > camarilla_r1_6h_aligned[i]) and vol_confirm and \
           (close[i] > ema_34_1d_aligned[i]) and (close[i] > ema_50_1w_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 6h Camarilla S1 level
        # 2. Volume confirmation
        # 3. Price below 1d EMA34 (short-term downtrend)
        # 4. Price below 1w EMA50 (long-term downtrend)
        elif (close[i] < camarilla_s1_6h_aligned[i]) and vol_confirm and \
             (close[i] < ema_34_1d_aligned[i]) and (close[i] < ema_50_1w_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R1S1_Volume_1dEMA34_1wEMA50_Filter_v1"
timeframe = "6h"
leverage = 1.0