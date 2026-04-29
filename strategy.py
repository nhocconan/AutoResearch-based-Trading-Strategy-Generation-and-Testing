#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d Donchian(20) breakout + weekly EMA50 trend filter + volume confirmation
# Donchian breakouts capture strong momentum moves; weekly EMA50 filters for higher-timeframe trend
# (above/below weekly EMA50); volume confirms breakout validity.
# Works in both bull and bear markets by trading breakouts in the direction of weekly trend.
# Target: 20-50 trades/year (80-200 total over 4 years).

name = "1d_Donchian20_WeeklyEMA50_Trend_VolumeConfirm_v1"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load HTF data ONCE before loop for 1w calculations
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate weekly EMA50 on 1w data
    weekly_close = df_1w['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, min_periods=50, adjust=False).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_1w, weekly_ema50)
    
    # Calculate Donchian channels (20-period) on 1d data
    highest_high = pd.Series(high).rolling(window=20, min_periods=20).max().values
    lowest_low = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * vol_ma_20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 20)  # warmup for weekly EMA50 and Donchian
    
    for i in range(start_idx, n):
        # Skip if HTF data not available
        if np.isnan(weekly_ema50_aligned[i]) or np.isnan(highest_high[i]) or np.isnan(lowest_low[i]):
            signals[i] = 0.0
            continue
            
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        curr_volume_confirm = volume_confirm[i]
        curr_weekly_ema50 = weekly_ema50_aligned[i]
        curr_highest_high = highest_high[i]
        curr_lowest_low = lowest_low[i]
        
        if position == 0:  # Flat - look for new entries
            # Long breakout: price closes above upper Donchian + above weekly EMA50 + volume
            if curr_close > curr_highest_high and curr_close > curr_weekly_ema50 and curr_volume_confirm:
                signals[i] = 0.25
                position = 1
            # Short breakout: price closes below lower Donchian + below weekly EMA50 + volume
            elif curr_close < curr_lowest_low and curr_close < curr_weekly_ema50 and curr_volume_confirm:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:  # Long position - exit when price closes below weekly EMA50
            if curr_close < curr_weekly_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:  # Short position - exit when price closes above weekly EMA50
            if curr_close > curr_weekly_ema50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals