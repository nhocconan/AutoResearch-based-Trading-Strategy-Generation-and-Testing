#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R1/S1 breakout with volume confirmation and 12h trend filter
# Long when price breaks above Camarilla R1 (1d) + volume > 1.3x avg + 12h EMA50 rising
# Short when price breaks below Camarilla S1 (1d) + volume > 1.3x avg + 12h EMA50 falling
# Uses discrete position sizing (0.25) to minimize fee churn
# Target: 15-30 trades/year per symbol to avoid fee drag while capturing institutional levels

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d and 12h HTF data once before loop
    df_1d = get_htf_data(prices, '1d')
    df_12h = get_htf_data(prices, '12h')
    if len(df_1d) < 30 or len(df_12h) < 30:
        return np.zeros(n)
    
    # === 1d Indicators: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    pivot = (high_1d + low_1d + close_1d) / 3.0
    range_1d = high_1d - low_1d
    camarilla_r1 = pivot + (range_1d * 1.1 / 12.0)
    camarilla_s1 = pivot - (range_1d * 1.1 / 12.0)
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # === 12h Indicators: EMA50 for trend filter ===
    close_12h = df_12h['close'].values
    ema_50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_50_12h)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(ema_50_12h_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R1
        # 2. Volume confirmation
        # 3. 12h EMA50 rising (trend up)
        if (close[i] > camarilla_r1_aligned[i]) and vol_confirm and (ema_50_12h_aligned[i] > ema_50_12h_aligned[i-1]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S1
        # 2. Volume confirmation
        # 3. 12h EMA50 falling (trend down)
        elif (close[i] < camarilla_s1_aligned[i]) and vol_confirm and (ema_50_12h_aligned[i] < ema_50_12h_aligned[i-1]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Camarilla_R1S1_Volume_12hEMA_Filter_v1"
timeframe = "4h"
leverage = 1.0