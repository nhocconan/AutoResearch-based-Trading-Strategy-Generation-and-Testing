#!/usr/bin/env python3
name = "12h_Donchian20_Breakout_1dTrend_Volume"
timeframe = "12h"
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
    
    # Load daily data ONCE before loop for trend filter and Donchian
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Daily Donchian(20) channels
    high_20 = pd.Series(df_1d['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1d['low']).rolling(window=20, min_periods=20).min().values
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close']).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily indicators to 12h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1d, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1d, low_20)
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Volume spike detection: 2-period average (1 day of 12h bars)
    vol_ma_2 = pd.Series(volume).rolling(window=2, min_periods=2).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(34, 20, 2)  # Wait for EMA, Donchian, and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_ma_2[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high with volume and daily uptrend
            vol_condition = volume[i] > vol_ma_2[i] * 1.8
            uptrend = ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1]
            
            if close[i] > high_20_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low with volume and daily downtrend
            elif close[i] < low_20_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price back below Donchian low or volume drops
            if close[i] < low_20_aligned[i] or volume[i] < vol_ma_2[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price back above Donchian high or volume drops
            if close[i] > high_20_aligned[i] or volume[i] < vol_ma_2[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

# Hypothesis: 12h Donchian(20) breakout with 1d trend and volume confirmation
# - Donchian(20) on daily timeframe provides robust support/resistance levels
# - Breakout above Donchian high with volume in daily uptrend = long opportunity
# - Breakdown below Donchian low with volume in daily downtrend = short opportunity
# - Volume spike (1.8x average) confirms institutional participation
# - Works in both bull (buy Donchian high breaks in uptrend) and bear (sell Donchian low breaks in downtrend)
# - Exit when price returns to Donchian low (for longs) or high (for shorts) or volume weakens
# - Position size 0.25 targets ~20-50 trades/year, avoiding fee drag
# - Uses actual daily Donchian levels (not intraday) for better stability
# - Daily trend filter reduces whipsaws vs using same timeframe
# - Designed to work in BOTH bull and bear markets via trend filter
# - Volume confirmation reduces false breakouts
# - Novel combination: Donchian(20) (1d) + trend (1d) + volume (12h) not recently tried in 12h timeframe
# - Aims for 50-150 total trades over 4 years (12-37/year) to stay within limits