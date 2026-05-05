#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h strategy using daily Williams %R mean reversion with 4h EMA21 trend filter and volume confirmation
# Long when Williams %R(14) crosses above -80 (oversold) AND price > 4h EMA21 AND volume > 1.5 * avg_volume(20) on 4h
# Short when Williams %R(14) crosses below -20 (overbought) AND price < 4h EMA21 AND volume > 1.5 * avg_volume(20) on 4h
# Exit when Williams %R crosses back below -50 (for longs) or above -50 (for shorts) OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 80-150 total trades over 4 years (20-38/year) for 4h timeframe
# Williams %R identifies overextended moves likely to revert
# 4h EMA21 filters for trend alignment to avoid choppy counter-trend trades
# Volume confirmation ensures breakout/participation strength
# Works in bull markets (mean reversion in uptrend) and bear markets (mean reversion in downtrend)

name = "4h_WilliamsR_MeanReversion_4hEMA21_VolumeConfirm"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data ONCE before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least one completed daily bar for Williams %R
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Williams %R(14) on daily: %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # Avoid division by zero
    
    # Align daily Williams %R to 4h timeframe (wait for completed daily bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 4h data ONCE before loop for EMA21 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 21:  # Need enough for EMA21
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA21
    close_4h_series = pd.Series(close_4h)
    ema21_4h = close_4h_series.ewm(span=21, adjust=False, min_periods=21).mean().values
    ema21_4h_aligned = align_htf_to_ltf(prices, df_4h, ema21_4h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 4h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema21_4h_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R crosses above -80 (from below), price above EMA21, volume confirmation, in session
            if (williams_r_aligned[i] > -80 and williams_r_aligned[i-1] <= -80 and 
                close[i] > ema21_4h_aligned[i] and volume_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R crosses below -20 (from above), price below EMA21, volume confirmation, in session
            elif (williams_r_aligned[i] < -20 and williams_r_aligned[i-1] >= -20 and 
                  close[i] < ema21_4h_aligned[i] and volume_confirm[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back below -50 OR volume drops below average
            if (williams_r_aligned[i] < -50 and williams_r_aligned[i-1] >= -50) or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back above -50 OR volume drops below average
            if (williams_r_aligned[i] > -50 and williams_r_aligned[i-1] <= -50) or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals