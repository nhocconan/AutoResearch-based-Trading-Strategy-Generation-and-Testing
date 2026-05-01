#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + weekly EMA50 trend filter + volume confirmation (>1.5x 20-bar MA)
# Donchian breakout captures momentum, weekly EMA50 filters trend direction (long above weekly EMA50, short below),
# volume confirms strength. Works in bull markets via breakouts above weekly EMA50 and in bear markets via short
# breakdowns below weekly EMA50. Discrete sizing (0.25) minimizes fee churn. Target: 30-100 total trades over 4 years (7-25/year).

name = "1d_Donchian20_WeeklyEMA50_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 1w HTF data for weekly EMA50 calculation
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 on 1w data
    weekly_ema50 = pd.Series(df_1w['close']).ewm(span=50, min_periods=50, adjust=False).mean().values
    
    # Align weekly EMA50 to 1d timeframe
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Donchian(20) channels on 1d data
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    
    # Volume confirmation: current volume > 1.5 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (volume_ma_20 * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(lookback, 20)  # Need 20 for Donchian and volume MA
    
    for i in range(start_idx, n):
        if np.isnan(weekly_ema50_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or np.isnan(volume_ma_20[i]):
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume = volume[i]
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        if position == 0:  # Flat - look for new entries
            # Long: price breaks above Donchian upper band, above weekly EMA50, and volume confirmation
            if curr_close > highest_high[i-1] and curr_close > weekly_ema50_aligned[i] and vol_confirm:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band, below weekly EMA50, and volume confirmation
            elif curr_close < lowest_low[i-1] and curr_close < weekly_ema50_aligned[i] and vol_confirm:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit on price breaking below Donchian lower band or below weekly EMA50
            if curr_close < lowest_low[i-1] or curr_close < weekly_ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position
            # Exit on price breaking above Donchian upper band or above weekly EMA50
            if curr_close > highest_high[i-1] or curr_close > weekly_ema50_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals