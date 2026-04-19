#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h weekly Donchian breakout with daily volume confirmation and ATR filter.
# Weekly Donchian channels identify major support/resistance levels, daily volume confirms breakout strength,
# ATR filter avoids entries during low volatility. Designed for low trade frequency (~15-25/year) 
# to perform in both bull and bear markets by capturing significant trend changes.
# Entry: Price breaks above weekly Donchian high OR below weekly Donchian low + daily volume spike + ATR > threshold.
# Exit: Price returns inside weekly Donchian channel.
name = "12h_WeeklyDonchian_DailyVol_ATR"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for Donchian channels
    df_weekly = get_htf_data(prices, '1w')
    weekly_high = df_weekly['high'].values
    weekly_low = df_weekly['low'].values
    
    # Calculate weekly Donchian channels (20-period)
    donchian_high = pd.Series(weekly_high).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(weekly_low).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_weekly, donchian_high)
    donchian_low_aligned = align_htf_to_ltf(prices, df_weekly, donchian_low)
    
    # Get daily data for volume and ATR
    df_daily = get_htf_data(prices, '1d')
    daily_volume = df_daily['volume'].values
    daily_high = df_daily['high'].values
    daily_low = df_daily['low'].values
    daily_close = df_daily['close'].values
    
    # Calculate daily volume moving average (20-period)
    volume_ma = pd.Series(daily_volume).rolling(window=20, min_periods=20).mean().values
    
    # Calculate daily ATR (14-period)
    tr1 = daily_high - daily_low
    tr2 = np.abs(daily_high - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr3 = np.abs(daily_low - np.concatenate([[daily_close[0]], daily_close[:-1]]))
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Align daily indicators to 12h timeframe
    volume_ma_aligned = align_htf_to_ltf(prices, df_daily, volume_ma)
    atr_aligned = align_htf_to_ltf(prices, df_daily, atr)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure enough data for all indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_aligned[i]) or np.isnan(donchian_low_aligned[i]) or
            np.isnan(volume_ma_aligned[i]) or np.isnan(atr_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above weekly Donchian high + volume spike + sufficient volatility
            if (close[i] > donchian_high_aligned[i] and 
                volume[i] > volume_ma_aligned[i] * 1.5 and 
                atr_aligned[i] > 0.01 * close[i]):  # ATR > 1% of price
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below weekly Donchian low + volume spike + sufficient volatility
            elif (close[i] < donchian_low_aligned[i] and 
                  volume[i] > volume_ma_aligned[i] * 1.5 and 
                  atr_aligned[i] > 0.01 * close[i]):
                signals[i] = -0.25
                position = -1
                
        elif position == 1:
            # Long: exit if price returns inside weekly Donchian channel
            if close[i] < donchian_high_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
                
        elif position == -1:
            # Short: exit if price returns inside weekly Donchian channel
            if close[i] > donchian_low_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals