#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Camarilla R1/S1 breakout with weekly pivot direction filter and volume confirmation
# Long when price breaks above Camarilla R1 (1d) + weekly pivot shows bullish bias (weekly close > weekly open) + volume > 1.5x 20-period avg
# Short when price breaks below Camarilla S1 (1d) + weekly pivot shows bearish bias (weekly close < weekly open) + volume > 1.5x 20-period avg
# Uses discrete position sizing (0.25) to control drawdown and minimize fee drag.
# Weekly pivot bias provides multi-timeframe alignment reducing whipsaws in both bull and bear markets.
# Volume threshold (1.5x) targets ~15-25 trades/year on 6h timeframe to avoid overtrading.
# Camarilla R1/S1 levels provide precise intraday breakout levels that work in ranging markets.

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
    
    # Get 1d HTF data once before loop for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # === 1d Indicator: Camarilla Pivot Levels (R1, S1) ===
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate pivot point
    pivot_1d = (high_1d + low_1d + close_1d) / 3.0
    # Calculate Camarilla levels
    range_1d = high_1d - low_1d
    camarilla_r1 = close_1d + range_1d * 1.1 / 12.0
    camarilla_s1 = close_1d - range_1d * 1.1 / 12.0
    
    # Align Camarilla levels to 6h timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    # Get 1w HTF data once before loop for weekly bias
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    # === 1w Indicator: Weekly Bias (bullish if close > open) ===
    open_1w = df_1w['open'].values
    close_1w = df_1w['close'].values
    weekly_bullish = close_1w > open_1w  # True for bullish weekly candle
    
    # Align weekly bias to 6h timeframe
    weekly_bullish_aligned = align_htf_to_ltf(prices, df_1w, weekly_bullish.astype(float))
    
    # Volume SMA for confirmation (using 20-period)
    vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = max(20, 20) + 5  # Camarilla + volume(20) + buffer
    
    for i in range(warmup, n):
        # Skip if outside trading session (08-20 UTC)
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r1_aligned[i]) or np.isnan(camarilla_s1_aligned[i]) or
            np.isnan(weekly_bullish_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period volume SMA
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.5)
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above Camarilla R1 (close > R1)
        # 2. Weekly pivot shows bullish bias (weekly close > weekly open)
        # 3. Volume confirmation
        if (close[i] > camarilla_r1_aligned[i]) and \
           weekly_bullish_aligned[i] > 0.5 and vol_confirm:
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below Camarilla S1 (close < S1)
        # 2. Weekly pivot shows bearish bias (weekly close < weekly open)
        # 3. Volume confirmation
        elif (close[i] < camarilla_s1_aligned[i]) and \
             weekly_bullish_aligned[i] < 0.5 and vol_confirm:
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "6h_Camarilla_R1S1_1d_1wBias_Volume_Filter_v1"
timeframe = "6h"
leverage = 1.0