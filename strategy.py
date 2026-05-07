#!/usr/bin/env python3
name = "1h_1dRSI_TrendFilter_VolumeBreak"
timeframe = "1h"
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
    
    # Load daily data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:
        return np.zeros(n)
    
    # Daily RSI(14) for trend filter
    delta = pd.Series(df_1d['close']).diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=14).mean()
    rs = gain / loss
    rsi_14 = 100 - (100 / (1 + rs))
    rsi_14_values = rsi_14.values
    rsi_14_aligned = align_htf_to_ltf(prices, df_1d, rsi_14_values)
    
    # 1h Donchian channel (20-period)
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # 1h volume spike detection: 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 14)  # Wait for indicators
    
    for i in range(start_idx, n):
        if np.isnan(rsi_14_aligned[i]) or np.isnan(high_20[i]) or np.isnan(low_20[i]) or np.isnan(vol_ma_20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: 08-20 UTC
        hour = pd.Timestamp(prices['open_time'].iloc[i]).hour
        if hour < 8 or hour > 20:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and daily RSI > 50 (uptrend)
            vol_condition = volume[i] > vol_ma_20[i] * 1.5
            if close[i] > high_20[i] and vol_condition and rsi_14_aligned[i] > 50:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Donchian low with volume and daily RSI < 50 (downtrend)
            elif close[i] < low_20[i] and vol_condition and rsi_14_aligned[i] < 50:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price returns to Donchian low or volume drops
            if close[i] < low_20[i] or volume[i] < vol_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price returns to Donchian high or volume drops
            if close[i] > high_20[i] or volume[i] < vol_ma_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Donchian breakout with daily RSI trend filter and volume confirmation
# - Daily RSI > 50 indicates uptrend, < 50 indicates downtrend (works in bull/bear)
# - Breakout above 20-period high with volume in uptrend = long
# - Breakdown below 20-period low with volume in downtrend = short
# - Volume spike (1.5x average) confirms institutional participation
# - Session filter (08-20 UTC) reduces noise trades
# - Exit when price returns to opposite Donchian band or volume weakens
# - Position size 0.20 targets 15-30 trades/year, avoiding fee drag
# - Donchian provides clear breakout levels, RSI provides trend filter