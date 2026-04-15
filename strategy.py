#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla pivot breakout (R1/S1) with 1d EMA trend filter and volume confirmation.
# Uses 1d EMA(50) for trend bias and Camarilla R1/S1 levels for entry timing.
# Includes volume filter (current volume > 1.3x 20-bar volume SMA) to avoid false breakouts.
# Designed for low trade frequency (12-37/year) on 12h timeframe to minimize fee drag.
# Works in bull/bear: 1d EMA avoids counter-trend trades, Camarilla captures institutional levels.

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
    
    # === 1d Indicators: EMA(50) for trend filter ===
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # === 1d Indicators: Camarilla pivot levels (R1, S1) ===
    # Camarilla: R1 = C + (H-L)*1.1/12, S1 = C - (H-L)*1.1/12
    close_1d = pd.Series(df_1d['close'].values)
    high_1d = pd.Series(df_1d['high'].values)
    low_1d = pd.Series(df_1d['low'].values)
    camarilla_r1 = (close_1d + (high_1d - low_1d) * 1.1 / 12).values
    camarilla_s1 = (close_1d - (high_1d - low_1d) * 1.1 / 12).values
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1
        # 2. 1d price above EMA50 (bullish trend bias)
        # 3. Volume confirmation
        if (close[i] > camarilla_r1_aligned[i] and
            close[i] > ema_50_1d_aligned[i] and
            vol_confirm):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1
        # 2. 1d price below EMA50 (bearish trend bias)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s1_aligned[i] and
              close[i] < ema_50_1d_aligned[i] and
              vol_confirm):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_EMA50_VolFilter_v1"
timeframe = "12h"
leverage = 1.0