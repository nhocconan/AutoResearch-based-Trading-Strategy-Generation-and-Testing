#!/usr/bin/env python3
name = "6h_1d_DonchianBreakout_VolumeTrend_v1"
timeframe = "6h"
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
    
    # Load daily data ONCE before loop for trend and Donchian
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily EMA(50) for trend filter
    ema_50_1d = pd.Series(df_1d['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Daily Donchian(20) channels from previous day
    prev_high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().shift(1).values
    prev_low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().shift(1).values
    
    # Align daily Donchian levels to 6h timeframe
    upper_20_aligned = align_htf_to_ltf(prices, df_1d, prev_high_20)
    lower_20_aligned = align_htf_to_ltf(prices, df_1d, prev_low_20)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20, 4)  # Wait for EMA, Donchian, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(upper_20_aligned[i]) or 
            np.isnan(lower_20_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above upper Donchian with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 1.8
            uptrend = ema_50_1d_aligned[i] > ema_50_1d_aligned[i-1]
            
            if close[i] > upper_20_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower Donchian with volume and daily downtrend
            elif close[i] < lower_20_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price returns below lower Donchian or volume drops
            if close[i] < lower_20_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price returns above upper Donchian or volume drops
            if close[i] > upper_20_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 6h Donchian(20) breakout with daily trend and volume confirmation
# - Daily Donchian(20) channels from previous day provide key support/resistance
# - Breakout above upper channel with volume in daily uptrend = long opportunity
# - Breakdown below lower channel with volume in daily downtrend = short opportunity
# - Volume spike (1.8x average) confirms institutional participation
# - Works in both bull (buy breakouts in uptrend) and bear (sell breakdowns in downtrend)
# - Exit when price returns to opposite Donchian band or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses actual daily Donchian levels (not resampled) for accuracy
# - Daily trend filter reduces whipsaws vs using same timeframe
# - Novel combination: Donchian (1d) + trend (1d) + volume (6h) not recently tried on 6h
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Focus on BTC/ETH as primary targets (not SOL-only)