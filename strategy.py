#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with volume confirmation and 1d EMA200 trend filter
# Long when price breaks above 20-period Donchian high + volume > 1.3x 20-period volume avg + price > 1d EMA200
# Short when price breaks below 20-period Donchian low + volume > 1.3x 20-period volume avg + price < 1d EMA200
# Uses Donchian channels for price structure and 1d EMA200 for multi-timeframe trend alignment
# Designed for low trade frequency (15-30/year) to minimize fee drag and improve test generalization
# Volume confirmation reduces false breakouts
# Works in both bull and bear markets by requiring volume confirmation and 1d trend alignment

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
    if len(df_1d) < 60:
        return np.zeros(n)
    
    # === 1d Indicator: EMA200 ===
    close_1d = df_1d['close'].values
    ema_200_1d = pd.Series(close_1d).ewm(span=200, adjust=False, min_periods=200).mean().values
    ema_200_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # === 4h Indicators: Donchian(20) ===
    # Calculate 20-period Donchian channels on 4h data
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels: 20-period high and low
    high_20 = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    
    signals = np.zeros(n)
    
    # Warmup: ensure all indicators are valid
    warmup = 100
    
    for i in range(warmup, n):
        # Volume filter: current volume > 1.3x 20-period volume SMA
        vol_sma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
        vol_confirm = volume[i] > (vol_sma_20[i] * 1.3)
        
        # Skip if any required data is NaN
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or
            np.isnan(ema_200_1d_aligned[i]) or np.isnan(vol_sma_20[i])):
            signals[i] = 0.0
            continue
        
        # === LONG CONDITIONS ===
        # 1. Price breaks above 4h Donchian(20) high
        # 2. Volume confirmation
        # 3. Price above 1d EMA200 (long-term uptrend)
        if (close[i] > high_20_aligned[i]) and vol_confirm and (close[i] > ema_200_1d_aligned[i]):
            signals[i] = 0.25
        
        # === SHORT CONDITIONS ===
        # 1. Price breaks below 4h Donchian(20) low
        # 2. Volume confirmation
        # 3. Price below 1d EMA200 (long-term downtrend)
        elif (close[i] < low_20_aligned[i]) and vol_confirm and (close[i] < ema_200_1d_aligned[i]):
            signals[i] = -0.25
        
        else:
            signals[i] = 0.0  # flat
    
    return signals

name = "4h_Donchian20_Volume_1dEMA200_Filter_v1"
timeframe = "4h"
leverage = 1.0