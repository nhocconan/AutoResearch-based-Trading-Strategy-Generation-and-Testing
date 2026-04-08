#!/usr/bin/env python3
# 4h_1d_1w_volume_spike_breakout
# Hypothesis: Trade 4-hour breakouts of daily Donchian channels with weekly trend filter and volume spike confirmation.
# Long when price breaks above daily Donchian(20) high with volume > 2x 20-period average and weekly close > weekly EMA50.
# Short when price breaks below daily Donchian(20) low with volume > 2x 20-period average and weekly close < weekly EMA50.
# Uses volume surge to filter false breakouts and weekly trend to align with higher timeframe momentum.
# Designed for 4h timeframe to target 20-50 trades/year (80-200 total over 4 years).
# Weekly trend filter works in both bull and bear markets by only trading in direction of higher timeframe trend.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "4h_1d_1w_volume_spike_breakout"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily data for Donchian channels
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Donchian channels: 20-period high/low
    donch_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donch_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Weekly data for trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    weekly_ema50 = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align daily and weekly data to 4h timeframe
    donch_high_aligned = align_htf_to_ltf(prices, df_1d, donch_high)
    donch_low_aligned = align_htf_to_ltf(prices, df_1d, donch_low)
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Volume confirmation: volume > 2x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    start_idx = 60  # Ensure all indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(donch_high_aligned[i]) or np.isnan(donch_low_aligned[i]) or 
            np.isnan(weekly_ema50_aligned[i]) or np.isnan(vol_ma_20[i])):
            if position != 0:
                pass  # Hold position
            else:
                signals[i] = 0.0
            continue
        
        # Volume spike condition
        vol_spike = volume[i] > 2.0 * vol_ma_20[i] if vol_ma_20[i] > 0 else False
        
        if position == 1:  # Long position
            # Exit: price breaks below daily Donchian low
            if close[i] < donch_low_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price breaks above daily Donchian high
            if close[i] > donch_high_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above daily Donchian high with volume spike and weekly uptrend
            if (close[i] > donch_high_aligned[i] and vol_spike and 
                close[i] > weekly_ema50_aligned[i]):
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below daily Donchian low with volume spike and weekly downtrend
            elif (close[i] < donch_low_aligned[i] and vol_spike and 
                  close[i] < weekly_ema50_aligned[i]):
                position = -1
                signals[i] = -0.25
    
    return signals