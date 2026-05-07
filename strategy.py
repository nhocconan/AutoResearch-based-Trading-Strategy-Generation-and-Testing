#!/usr/bin/env python3
name = "4h_Donchian20_12hTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 12h data ONCE for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Calculate 12h EMA(20) for trend filter
    ema_20_12h = pd.Series(df_12h['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_20_12h)
    
    # Donchian(20) channels on 4h
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume spike detection: 4-period average (1 day of 4h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 4)
    
    for i in range(start_idx, n):
        if (np.isnan(ema_20_12h_aligned[i]) or np.isnan(high_20[i]) or 
            np.isnan(low_20[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price above upper band with volume and 12h uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            uptrend = ema_20_12h_aligned[i] > ema_20_12h_aligned[i-1]
            
            if close[i] > high_20[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price below lower band with volume and 12h downtrend
            elif close[i] < low_20[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below upper band or volume drops
            if close[i] < high_20[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above lower band or volume drops
            if close[i] > low_20[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: Donchian(20) breakout on 4h with 12h trend filter and volume confirmation
# - Long when price breaks above 20-period high with volume spike in 12h uptrend
# - Short when price breaks below 20-period low with volume spike in 12h downtrend
# - Volume spike (1.8x average) confirms institutional participation
# - Exit when price returns to the breakout level or volume weakens
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - Works in both bull and bear markets via 12h trend filter
# - Volume confirmation reduces false breakouts
# - Simple 3-condition strategy avoids overtrading and curve-fitting
# - Uses actual Donchian channels (not SMA/EMA) for clear breakout levels
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits