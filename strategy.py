#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla pivot breakout with volume confirmation and session filter.
# Uses 4h Camarilla R1/S1 levels for entry timing, filtered by 1d EMA50 trend and volume spike.
# Session filter (08-20 UTC) reduces noise trades. Designed for low trade frequency (15-35/year)
# to minimize fee drag. Works in bull/bear: 1d EMA50 avoids counter-trend trades, Camarilla
# breakouts capture institutional level reactions with momentum confirmation.

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    
    # Get 4h and 1d HTF data once before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 50 or len(df_1d) < 50:
        return np.zeros(n)
    
    # === 4h Indicators: Camarilla Pivots (R1, S1) ===
    # Camarilla formula: Close +- (High-Low) * 1.1/12
    close_4h = df_4h['close'].values
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    camarilla_high = close_4h + (high_4h - low_4h) * 1.1 / 12  # R1 level
    camarilla_low = close_4h - (high_4h - low_4h) * 1.1 / 12   # S1 level
    
    camarilla_high_aligned = align_htf_to_ltf(prices, df_4h, camarilla_high)
    camarilla_low_aligned = align_htf_to_ltf(prices, df_4h, camarilla_low)
    
    # === 1d Indicators: Trend Filter ===
    # 1d EMA(50) for trend bias
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Session filter: 08-20 UTC only
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
            
        # Volume filter: current volume > 2.0x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 2.0)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_high_aligned[i]) or np.isnan(camarilla_low_aligned[i]) or
            np.isnan(ema_50_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1 (4h resistance)
        # 2. 1d price above EMA50 (bullish trend bias)
        # 3. Volume confirmation
        if (close[i] > camarilla_high_aligned[i] and
            close[i] > ema_50_1d_aligned[i] and
            vol_confirm):
            signals[i] = 0.20
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1 (4h support)
        # 2. 1d price below EMA50 (bearish trend bias)
        # 3. Volume confirmation
        elif (close[i] < camarilla_low_aligned[i] and
              close[i] < ema_50_1d_aligned[i] and
              vol_confirm):
            signals[i] = -0.20
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1h_Camarilla_R1S1_EMA50_VolFilter_v1"
timeframe = "1h"
leverage = 1.0