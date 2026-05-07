#!/usr/bin/env python3
name = "1h_4h_Donchian_Breakout_1dTrend_Volume"
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
    
    # Load 4h data ONCE for Donchian breakout levels
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:
        return np.zeros(n)
    
    # Load 1d data ONCE for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # 4h Donchian channels (20-period) for breakout detection
    donchian_high = pd.Series(df_4h['high']).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(df_4h['low']).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 1h timeframe (waits for 4h bar close)
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low)
    
    # 1d EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # 1h volume spike detection: 24-period average (1 day of 1h bars)
    vol_ma_24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 24)  # Wait for EMA, Donchian, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_24[i])):
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
            # Long: break above 4h Donchian high with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_24[i] * 2.0
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if close[i] > donchian_high_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.20
                position = 1
            # Short: break below 4h Donchian low with volume and daily downtrend
            elif close[i] < donchian_low_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit: price back below 4h Donchian low or volume drops
            if close[i] < donchian_low_aligned[i] or volume[i] < vol_ma_24[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit: price back above 4h Donchian high or volume drops
            if close[i] > donchian_high_aligned[i] or volume[i] < vol_ma_24[i] * 1.5:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

# Hypothesis: 1h Donchian breakout with 4h structure and 1d trend filter
# - Uses 4h Donchian channels (20-period) as primary structure for breakouts
# - 1d EMA(50) filter ensures trades align with higher timeframe trend
# - Volume confirmation (2.0x average) filters false breakouts
# - Session filter (08-20 UTC) reduces noise during low-activity periods
# - Designed for both bull and bear markets via trend filter
# - Position size 0.20 manages risk in volatile markets
# - Target: 15-35 trades/year to avoid fee drag (60-140 total over 4 years)
# - Exit when price returns to opposite Donchian band or volume weakens
# - Novel combination: 4h structure + 1d trend + 1h timing with volume confirmation