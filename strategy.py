#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with volume confirmation and trend filter (1d EMA50) works in both bull and bear markets.
# Uses breakout structure + volume confirmation to avoid false breakouts, with trend filter to align with higher timeframe momentum.
# Designed for moderate trade frequency (~25-40/year) to minimize fee drag while capturing sustained moves.
name = "4h_Donchian20_1dEMA50_Volume_Trend"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend and volume filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Trend filter: 1d EMA50
    ema50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_series = pd.Series(volume)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_ma * 1.5)
    
    # Align 1d EMA50 to 4h
    ema50_1d_4h = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup period for Donchian and EMA
    start_idx = max(20, 50)
    
    for i in range(start_idx, n):
        if np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or np.isnan(ema50_1d_4h[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: break above Donchian high with volume and above 1d EMA50
            if close[i] > donchian_high[i] and volume_filter[i] and close[i] > ema50_1d_4h[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: break below Donchian low with volume and below 1d EMA50
            elif close[i] < donchian_low[i] and volume_filter[i] and close[i] < ema50_1d_4h[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: close below Donchian low (breakdown of structure)
            if close[i] < donchian_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: close above Donchian high (breakout of structure)
            if close[i] > donchian_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals