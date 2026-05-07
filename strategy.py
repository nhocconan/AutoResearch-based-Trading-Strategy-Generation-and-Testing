#!/usr/bin/env python3
name = "4h_Donchian20_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE before loop for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(20) for trend filter
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Calculate Donchian channels (20-period) on 4h data
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: 2-period average (previous 8h)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 2)  # Wait for Donchian and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume and 12h uptrend
            vol_condition = volume[i] > vol_ma_2[i] * 1.5
            uptrend = ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1]
            
            if close[i] > high_20[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume and 12h downtrend
            elif close[i] < low_20[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below lower Donchian or volume drops
            if close[i] < low_20[i] or volume[i] < vol_ma_2[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above upper Donchian or volume drops
            if close[i] > high_20[i] or volume[i] < vol_ma_2[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian(20) breakout with 12h trend and volume confirmation
# - Donchian(20) breakout captures momentum in trending markets
# - 12h EMA(20) trend filter ensures we trade in the direction of higher timeframe trend
# - Volume spike (1.5x average) confirms institutional participation and reduces false breakouts
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Exit when price reverses to opposite Donchian band or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding excessive fee drag
# - Uses actual 4h and 12h data via mtf_data helper to avoid look-ahead
# - Designed for BTC/ETH with proper risk management via trend filter and volume confirmation