#!/usr/bin/env python3
"""
1d_Donchian_Breakout_WeeklyTrend_Volume
Hypothesis: Breaking above weekly Donchian high (20-period) or below weekly Donchian low with daily price above/below weekly EMA20 trend and volume spike (1.5x average) captures strong weekly momentum. Weekly trend filter reduces whipsaw in choppy markets while maintaining sufficient trades. Designed for 1d to achieve 15-25 trades/year with high win rate, suitable for both bull and bear markets by following higher timeframe trend.
"""
name = "1d_Donchian_Breakout_WeeklyTrend_Volume"
timeframe = "1d"
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
    
    # Get weekly data for trend filter and Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 2:
        return np.zeros(n)
    
    # Weekly Donchian channels (20-period)
    high_weekly = df_weekly['high'].values
    low_weekly = df_weekly['low'].values
    donchian_high = pd.Series(high_weekly).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_weekly).rolling(window=20, min_periods=20).min().values
    
    # Weekly EMA20 for trend filter
    ema_20_weekly = pd.Series(df_weekly['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Align weekly indicators to daily timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    ema_20_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_20_weekly)
    
    # Volume filter: current volume > 1.5 * 20-period average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 1.5)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Need sufficient warmup for weekly indicators
    
    for i in range(start_idx, n):
        # Skip if any data is not ready
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or 
            np.isnan(ema_20_weekly_aligned[i]) or np.isnan(vol_avg[i]) or 
            np.isnan(volume_filter[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above weekly Donchian high + weekly uptrend + volume spike
            if (close[i] > donchian_high_aligned[i] and 
                close[i] > ema_20_weekly_aligned[i] and 
                volume_filter[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below weekly Donchian low + weekly downtrend + volume spike
            elif (close[i] < donchian_low_aligned[i] and 
                  close[i] < ema_20_weekly_aligned[i] and 
                  volume_filter[i]):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price returns to opposite Donchian level
            if position == 1:
                if close[i] <= donchian_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] >= donchian_high_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals