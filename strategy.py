#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot R1/S1 breakout with 1d EMA50 trend filter and volume confirmation.
# Uses 1d EMA(50) for trend bias and Camarilla levels from 1d for precise entry/exit.
# Volume filter (current volume > 1.3x 20-bar SMA) ensures breakouts have momentum.
# Designed for low trade frequency (12-37/year) to minimize fee drag in ranging markets.
# Works in bull/bear: 1d EMA avoids counter-trend trades, Camarilla provides institutional levels.

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
    
    # === 1d Indicators: EMA50 for trend, Camarilla Pivots (R1, S1) ===
    close_1d = pd.Series(df_1d['close'].values)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    
    # EMA50 for trend bias
    ema_50_1d = close_1d.ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Camarilla pivot levels (based on previous day)
    # R1 = close + 1.1*(high-low)/12
    # S1 = close - 1.1*(high-low)/12
    camarilla_range = (high_1d - low_1d)
    camarilla_r1 = close_1d + (camarilla_range * 1.1 / 12)
    camarilla_s1 = close_1d - (camarilla_range * 1.1 / 12)
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1.values)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1.values)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1 (bullish breakout)
        # 2. 1d price above EMA50 (bullish trend bias)
        # 3. Volume confirmation
        if (close[i] > camarilla_r1_aligned[i] and
            close[i] > ema_50_1d_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1 (bearish breakdown)
        # 2. 1d price below EMA50 (bearish trend bias)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s1_aligned[i] and
              close[i] < ema_50_1d_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_EMA50_VolumeFilter_v1"
timeframe = "12h"
leverage = 1.0