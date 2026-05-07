#!/usr/bin/env python3
name = "12h_Donchian_Breakout_TrendVolume_v1"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load daily data ONCE before loop for Donchian and trend
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Load 12h data ONCE before loop for volume confirmation
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 20:
        return np.zeros(n)
    
    # Donchian channels: 20-day high/low from previous day
    donch_high = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    donch_low = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # 12h EMA(34) for trend filter
    ema_34_12h = pd.Series(df_12h['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Volume spike detection: 4-period average (1 day of 12h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    # Align HTF data to 12h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(20, 34, 4)  # Wait for Donchian, EMA, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(ema_34_12h_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and 12h uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            uptrend = ema_34_12h_aligned[i] > ema_34_12h_aligned[i-1]
            
            if close[i] > donch_high_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and 12h downtrend
            elif close[i] < donch_low_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian low or volume drops
            if close[i] < donch_low_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian high or volume drops
            if close[i] > donch_high_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Donchian(20) breakout with 12h trend and volume confirmation
# - Donchian channels from daily timeframe provide robust support/resistance
# - Breakout above 20-day high with volume in 12h uptrend = long opportunity
# - Breakdown below 20-day low with volume in 12h downtrend = short opportunity
# - Volume spike (1.8x average) confirms institutional participation
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Exit when price returns to opposite Donchian band or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses actual daily Donchian levels (not intraday) for better reliability
# - 12h trend filter reduces whipsaws vs using same timeframe
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Novel combination: Donchian (1d) + trend (12h) + volume (12h) not recently tried on 12h
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits