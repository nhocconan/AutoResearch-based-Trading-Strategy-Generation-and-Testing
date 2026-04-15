#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla R1/S1 breakout with volume confirmation and 1w EMA trend filter
# Long when price breaks above Camarilla R1 (1d) + volume > 1.8x 20-period avg + price > 1w EMA50
# Short when price breaks below Camarilla S1 (1d) + volume > 1.8x 20-period avg + price < 1w EMA50
# Uses 1d Camarilla levels and 1w EMA for trend alignment to reduce false breakouts
# Designed for low trade frequency (12-30/year) to minimize fee drag while capturing institutional levels

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 1w HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_1w = get_htf_data(prices, '1w')
    if len(df_1d) < 30 or len(df_1w) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla calculations
    range_1d = high_1d - low_1d
    camarilla_pivot = (high_1d + low_1d + close_1d) / 3
    camarilla_r1 = camarilla_pivot + (range_1d * 1.1 / 12)
    camarilla_s1 = camarilla_pivot - (range_1d * 1.1 / 12)
    
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 1w Indicators: EMA50 for Trend Filter ===
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.8x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.8)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R1
        # 2. Volume confirmation
        # 3. Price above 1w EMA50 (uptrend filter)
        if (close[i] > camarilla_r1_aligned[i]) and vol_confirm and (close[i] > ema_50_1w_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S1
        # 2. Volume confirmation
        # 3. Price below 1w EMA50 (downtrend filter)
        elif (close[i] < camarilla_s1_aligned[i]) and vol_confirm and (close[i] < ema_50_1w_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "12h_Camarilla_R1S1_Volume_1wEMA50_Filter_v1"
timeframe = "12h"
leverage = 1.0