#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h strategy using 1d Williams %R extremes with 1d EMA34 trend filter and volume confirmation
# Long when 1d Williams %R < -80 (oversold) AND 1d EMA34 > EMA89 AND volume > 2.0 * avg_volume(20)
# Short when 1d Williams %R > -20 (overbought) AND 1d EMA34 < EMA89 AND volume > 2.0 * avg_volume(20)
# Exit when Williams %R crosses back above -50 (for long) or below -50 (for short)
# Uses discrete sizing 0.25 to balance return and drawdown control
# Target: 50-150 total trades over 4 years (12-37/year) for 6h timeframe
# Williams %R identifies exhaustion points that work in both bull (buy panic) and bear (sell euphoria)
# 1d EMA34/EMA89 filter ensures alignment with intermediate trend to avoid counter-trend trades
# Volume confirmation filters weak signals and ensures participation

name = "6h_1dWilliamsRExtreme_1dEMA34Trend_Volume_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data ONCE before loop for Williams %R and EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 100:  # Need sufficient data for EMA89
        return np.zeros(n)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d Williams %R (14-period)
    # Williams %R = (Highest High - Close) / (Highest High - Lowest Low) * -100
    highest_high_14 = pd.Series(high_1d).rolling(window=14, min_periods=14).max().values
    lowest_low_14 = pd.Series(low_1d).rolling(window=14, min_periods=14).min().values
    williams_r = (highest_high_14 - close_1d) / (highest_high_14 - lowest_low_14) * -100
    # Handle division by zero when high == low
    williams_r = np.where((highest_high_14 - lowest_low_14) == 0, -50, williams_r)
    
    # Calculate 1d EMA34 and EMA89 for trend filter
    close_series_1d = pd.Series(close_1d)
    ema_34_1d = close_series_1d.ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_89_1d = close_series_1d.ewm(span=89, adjust=False, min_periods=89).mean().values
    
    # Calculate volume confirmation: volume > 2.0 * 20-period average volume
    avg_volume_20 = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume_1d > (2.0 * avg_volume_20)
    
    # Align 1d indicators to 6h timeframe (wait for completed 1d bar)
    williams_r_aligned = align_htf_to_ltf(prices, df_1d, williams_r)
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    ema_89_aligned = align_htf_to_ltf(prices, df_1d, ema_89_1d)
    volume_confirm_aligned = align_htf_to_ltf(prices, df_1d, volume_confirm.astype(float))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(williams_r_aligned[i]) or np.isnan(ema_34_aligned[i]) or 
            np.isnan(ema_89_aligned[i]) or np.isnan(volume_confirm_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: Williams %R < -80 (oversold) with EMA34 > EMA89 and volume confirmation
            if (williams_r_aligned[i] < -80 and williams_r_aligned[i-1] >= -80 and 
                ema_34_aligned[i] > ema_89_aligned[i] and volume_confirm_aligned[i] > 0.5):
                signals[i] = 0.25
                position = 1
            # Short: Williams %R > -20 (overbought) with EMA34 < EMA89 and volume confirmation
            elif (williams_r_aligned[i] > -20 and williams_r_aligned[i-1] <= -20 and 
                  ema_34_aligned[i] < ema_89_aligned[i] and volume_confirm_aligned[i] > 0.5):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: Williams %R crosses back above -50 (recovering from oversold)
            if williams_r_aligned[i] > -50 and williams_r_aligned[i-1] <= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: Williams %R crosses back below -50 (declining from overbought)
            if williams_r_aligned[i] < -50 and williams_r_aligned[i-1] >= -50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals