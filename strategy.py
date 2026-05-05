#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using 1d Williams %R extreme reversal with 12h EMA34 trend filter and volume spike confirmation
# Long when Williams %R < -80 (oversold) AND price > 12h EMA34 AND volume > 1.5 * avg_volume(20) on 12h
# Short when Williams %R > -20 (overbought) AND price < 12h EMA34 AND volume > 1.5 * avg_volume(20) on 12h
# Exit when Williams %R crosses back above -50 (for long) or below -50 (for short) OR volume drops below average
# Uses discrete sizing 0.25 to balance return and risk
# Target: 50-150 total trades over 4 years (12-37/year) for 12h timeframe
# Williams %R identifies exhaustion points in both bull and bear markets
# 12h EMA34 filters for primary trend to avoid counter-trend trades
# Volume confirmation reduces false signals from weak breakouts
# Works in ranging markets (mean reversion from extremes) and trending markets (pullbacks in trend)

name = "12h_WilliamsR_Extreme_12hEMA34_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 14:  # Need at least one completed 1d bar for Williams %R
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)  # avoid division by zero
    
    # Align 1d Williams %R to 12h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    
    # Get 12h data ONCE before loop for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:  # Need enough for EMA34
        return np.zeros(n)
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA34
    close_12h_series = pd.Series(close_12h)
    ema34_12h = close_12h_series.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema34_12h)
    
    # Calculate volume confirmation: volume > 1.5 * 20-period average volume on 12h
    avg_volume_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > (1.5 * avg_volume_20)
    
    # Session filter: 08-20 UTC (pre-compute for efficiency)
    hours = prices.index.hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN or outside session
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema34_12h_aligned[i]) or 
            np.isnan(avg_volume_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold), price > 12h EMA34, volume confirmation, in session
            if williams_r_aligned[i] < -80 and close[i] > ema34_12h_aligned[i] and volume_confirm[i]:
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought), price < 12h EMA34, volume confirmation, in session
            elif williams_r_aligned[i] > -20 and close[i] < ema34_12h_aligned[i] and volume_confirm[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses above -50 OR volume drops below average
            if williams_r_aligned[i] > -50 or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses below -50 OR volume drops below average
            if williams_r_aligned[i] < -50 or volume[i] < avg_volume_20[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals