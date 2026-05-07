#!/usr/bin/env python3
name = "4h_Donchian_Breakout_VolumeTrend_v1"
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
    
    # Load daily data ONCE for Donchian and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Daily Donchian(20): highest high and lowest low of past 20 days
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Align daily Donchian levels to 4h timeframe
    donchian_high = align_htf_to_ltf(prices, df_1d, high_20)
    donchian_low = align_htf_to_ltf(prices, df_1d, low_20)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume spike detection: 6-period average (1.5 days of 4h bars)
    vol_ma_6 = pd.Series(volume).rolling(window=6, min_periods=6).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 6)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ma_6[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: break above Donchian high with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_6[i] * 2.0
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if close[i] > donchian_high[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: break below Donchian low with volume and daily downtrend
            elif close[i] < donchian_low[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian low or volume drops
            if close[i] < donchian_low[i] or volume[i] < vol_ma_6[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian high or volume drops
            if close[i] > donchian_high[i] or volume[i] < vol_ma_6[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 4h Donchian(20) breakout from daily levels with volume confirmation and daily trend filter
# - Uses actual daily Donchian channels (20-day high/low) as institutional support/resistance
# - Breakout above daily Donchian high with volume spike in daily uptrend = long
# - Breakdown below daily Donchian low with volume spike in daily downtrend = short
# - Volume spike (2.0x 6-period average) confirms institutional participation
# - Daily EMA(50) trend filter prevents counter-trend trades
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Position size 0.25 targets ~20-50 trades/year to avoid fee drag
# - Exit when price returns to opposite Donchian band or volume weakens
# - Uses 4h timeframe for execution while relying on daily structure for signal quality
# - Aims for 80-200 total trades over 4 years (20-50/year) within profitable range