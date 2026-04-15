#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Camarilla R1/S1 breakout with volume confirmation and weekly trend filter
# Long when price breaks above 1d Camarilla R1 + volume > 1.3x avg + weekly close > weekly open (bullish weekly bias)
# Short when price breaks below 1d Camarilla S1 + volume > 1.3x avg + weekly close < weekly open (bearish weekly bias)
# Uses discrete position sizing (0.25) to minimize fee drag, targeting 20-50 trades/year
# Weekly trend filter reduces false breakouts in counter-trend markets

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
    if len(df_1d) < 30 or len(df_1w) < 10:
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
    
    # === 1w Indicators: Weekly Trend (bullish/bearish bias) ===
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    open_1w = df_1w['open'].values
    
    weekly_bullish = close_1w > open_1w  # True for bullish weekly candle
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 1d Camarilla R1
        # 2. Volume confirmation
        # 3. Weekly bullish bias (close > open)
        if (close[i] > camarilla_r1_aligned[i]) and vol_confirm and weekly_bullish_aligned[i]:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 1d Camarilla S1
        # 2. Volume confirmation
        # 3. Weekly bearish bias (close < open)
        elif (close[i] < camarilla_s1_aligned[i]) and vol_confirm and (not weekly_bullish_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "1d_Camarilla_R1S1_Volume_WeeklyTrend_v1"
timeframe = "1d"
leverage = 1.0