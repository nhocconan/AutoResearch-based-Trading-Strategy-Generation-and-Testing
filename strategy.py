#!/usr/bin/env python3
name = "1h_4h_Donchian_1dTrend_Volume"
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
    
    # Load 4h data ONCE before loop for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Load 1d data ONCE before loop for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # 4h Donchian channel (20-period)
    high_20 = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_4h, high_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_4h, low_20)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection: 24-period average (1 day of 1h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 24)  # Wait for EMA, Donchian, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(vol_ma_24[i])):
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
            # Long: price breaks above upper Donchian with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 1.5
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if close[i] > upper_20_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below lower Donchian with volume and daily downtrend
            elif close[i] < lower_20_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price returns below lower Donchian or volume drops
            if close[i] < lower_20_aligned[i] or volume[i] < vol_ma_24[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price returns above upper Donchian or volume drops
            if close[i] > upper_20_aligned[i] or volume[i] < vol_ma_24[i] * 0.8:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Donchian(20) breakout with 1d trend and volume confirmation
# - Uses 4h Donchian channels for structure (reduces noise vs 1h Donchian)
# - 1d EMA(50) filter ensures trades align with daily trend
# - Volume spike (1.5x 24-bar average) confirms institutional participation
# - Session filter (08-20 UTC) reduces noise from low-liquidity hours
# - Position size 0.20 limits drawdown during 2022-like crashes
# - Designed to work in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Target: 60-150 total trades over 4 years (15-37/year) to stay within fee limits
# - Exit when price returns to opposite Donchian band or volume weakens
# - 4h timeframe for structure avoids whipsaws from pure 1h signals