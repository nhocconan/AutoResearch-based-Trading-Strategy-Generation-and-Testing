#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and 1w EMA50 trend filter
# Long when price breaks above 1w Camarilla R1 level + volume > 1.3x 20-period avg + price > 1w EMA50
# Short when price breaks below 1w Camarilla S1 level + volume > 1.3x 20-period avg + price < 1w EMA50
# Uses 1w price structure (Camarilla pivots) and 1w EMA50 for multi-timeframe trend alignment on 12h chart
# Designed for low trade frequency (12-25/year) to minimize fee drag and improve test generalization
# Volume confirmation reduces false breakouts, EMA50 filter ensures trend alignment

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w HTF data once before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 60:
        return np.zeros(n)
    
    # === 1w Indicators: Camarilla Pivot Levels (R1, S1) and EMA50 ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate pivot point (PP)
    pivot_point_1w = (high_1w + low_1w + close_1w) / 3.0
    
    # Calculate Camarilla levels (R1 and S1)
    camarilla_r1_1w = pivot_point_1w + (high_1w - low_1w) * 1.1 / 12.0
    camarilla_s1_1w = pivot_point_1w - (high_1w - low_1w) * 1.1 / 12.0
    
    # Calculate 1w EMA50
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align all HTF indicators to 12h timeframe
    camarilla_r1_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_r1_1w)
    camarilla_s1_1w_aligned = align_htf_to_ltf(prices, df_1w, camarilla_s1_1w)
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_1w_aligned[i]) or np.isnan(camarilla_s1_1w_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1w Camarilla R1 level
        # 2. Volume confirmation
        # 3. Price above 1w EMA50 (uptrend)
        if (close[i] > camarilla_r1_1w_aligned[i]) and vol_confirm and \
           (close[i] > ema_50_1w_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1w Camarilla S1 level
        # 2. Volume confirmation
        # 3. Price below 1w EMA50 (downtrend)
        elif (close[i] < camarilla_s1_1w_aligned[i]) and vol_confirm and \
             (close[i] < ema_50_1w_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_Volume_1wEMA50_Filter_v1"
timeframe = "12h"
leverage = 1.0