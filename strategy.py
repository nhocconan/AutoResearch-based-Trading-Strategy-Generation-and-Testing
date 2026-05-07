#!/usr/bin/env python3
name = "6h_1w_Donchian20_Breakout_Trend_Volume"
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
    
    # Load weekly data ONCE before loop
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Calculate weekly Donchian channels (20-period high/low)
    high_20 = pd.Series(df_1w['high']).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(df_1w['low']).rolling(window=20, min_periods=20).min().values
    
    # Align weekly Donchian channels to 6h timeframe
    high_20_aligned = align_htf_to_ltf(prices, df_1w, high_20)
    low_20_aligned = align_htf_to_ltf(prices, df_1w, low_20)
    
    # Weekly trend filter: EMA(50) on weekly close
    ema_50_1w = pd.Series(df_1w['close']).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_50_1w)
    
    # Volume spike detection: 4-period average (1 day of 6h bars)
    vol_ma_4 = pd.Series(volume).rolling(window=4, min_periods=4).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 4)  # Wait for EMA and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(high_20_aligned[i]) or np.isnan(low_20_aligned[i]) or 
            np.isnan(ema_50_1w_aligned[i]) or np.isnan(vol_ma_4[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high with volume and weekly uptrend
            vol_condition = volume[i] > vol_ma_4[i] * 2.0
            uptrend = ema_50_1w_aligned[i] > ema_50_1w_aligned[i-1]
            
            if close[i] > high_20_aligned[i] and vol_condition and uptrend:
                signals[i] = 0.30
                position = 1
            # Short: price breaks below weekly Donchian low with volume and weekly downtrend
            elif close[i] < low_20_aligned[i] and vol_condition and not uptrend:
                signals[i] = -0.30
                position = -1
        elif position == 1:
            # Exit: price returns below weekly Donchian high or volume drops
            if close[i] < high_20_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.30
        elif position == -1:
            # Exit: price returns above weekly Donchian low or volume drops
            if close[i] > low_20_aligned[i] or volume[i] < vol_ma_4[i] * 1.2:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.30
    
    return signals

# Hypothesis: 6h weekly Donchian breakout with volume and trend filter
# - Weekly Donchian channels (20-period) capture major support/resistance
# - Breakout above weekly high with volume in weekly uptrend = long opportunity
# - Breakdown below weekly low with volume in weekly downtrend = short opportunity
# - Volume spike (2.0x average) confirms institutional participation
# - Works in both bull (buy weekly high breaks in uptrend) and bear (sell weekly low breaks in downtrend)
# - Exit when price returns to weekly channel or volume weakens
# - Position size 0.30 targets ~25-50 trades/year, avoiding fee drag
# - Uses actual weekly data (not resampled) for correct alignment
# - Designed to work in BOTH bull and bear markets via trend filter
# - Weekly timeframe reduces noise and focuses on major trends
# - 6h timeframe allows timely entry/exit while keeping trade frequency low
# - Volume confirmation reduces false breakouts
# - Weekly EMA(50) ensures trading with the major trend direction
# - Target: 50-150 total trades over 4 years = 12-37/year
# - Position size 0.30 balances risk and return (max 30% loss in 2022-like crash)