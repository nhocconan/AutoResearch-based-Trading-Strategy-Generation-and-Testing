#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with 12h EMA trend filter and volume spike confirmation.
# Enter long when price breaks above Camarilla R3 level and 12h EMA50 > EMA200 and volume > 1.5x 20-bar average.
# Enter short when price breaks below Camarilla S3 level and 12h EMA50 < EMA200 and volume > 1.5x 20-bar average.
# Exit when price crosses Camarilla pivot point (PP) or 12h EMA50/EMA200 crossover reverses.
# Uses discrete position sizing (0.25) to minimize fee churn.
# Target: 80-150 total trades over 4 years (20-38/year) to avoid fee drag.
# Camarilla levels provide intraday support/resistance, EMA filters for trend alignment, volume confirms conviction.
# Works in both bull and bear markets by trading with the intermediate-term trend.

name = "4h_Camarilla_R3S3_Breakout_12hEMA_VolumeSpike_v1"
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
    
    # Get 12h data for EMA trend filter (MTF structure)
    df_12h = get_htf_data(prices, '12h')
    
    if len(df_12h) < 50:
        return np.zeros(n)
    
    # Calculate 12h EMA50 and EMA200
    close_12h = df_12h['close'].values
    ema_50 = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_200 = pd.Series(close_12h).ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align 12h EMAs to 4h timeframe
    ema_50_aligned = align_htf_to_ltf(prices, df_12h, ema_50)
    ema_200_aligned = align_htf_to_ltf(prices, df_12h, ema_200)
    
    # Calculate Camarilla levels from previous 1d bar (using get_htf_data for 1d)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Get previous 1d OHLC (Camarilla uses previous day's range)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # R4 = close + ((high-low)*1.1/2)
    # R3 = close + ((high-low)*1.1/4)
    # S3 = close - ((high-low)*1.1/4)
    # PP = (high + low + close)/3
    rng = high_1d - low_1d
    r3 = close_1d + (rng * 1.1 / 4)
    s3 = close_1d - (rng * 1.1 / 4)
    pp = (high_1d + low_1d + close_1d) / 3
    
    # Align Camarilla levels to 4h timeframe (use previous completed 1d bar)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    pp_aligned = align_htf_to_ltf(prices, df_1d, pp)
    
    # Volume confirmation: >1.5x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 1.5 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(100, 50)  # Ensure sufficient history for EMAs
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_50_aligned[i]) or np.isnan(ema_200_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or np.isnan(pp_aligned[i]) or
            np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Volume confirmation
        vol_confirm = volume_confirm[i]
        
        # Trend filter: 12h EMA50 > EMA200 for long, < for short
        uptrend = ema_50_aligned[i] > ema_200_aligned[i]
        downtrend = ema_50_aligned[i] < ema_200_aligned[i]
        
        # Camarilla breakout conditions
        breakout_up = close[i] > r3_aligned[i]  # Break above R3
        breakout_down = close[i] < s3_aligned[i]  # Break below S3
        
        # Exit conditions
        exit_long = close[i] < pp_aligned[i] or (uptrend and not downtrend and ema_50_aligned[i] < ema_200_aligned[i])
        exit_short = close[i] > pp_aligned[i] or (downtrend and not uptrend and ema_50_aligned[i] > ema_200_aligned[i])
        
        # Handle entries and exits
        if breakout_up and uptrend and vol_confirm and position <= 0:
            signals[i] = 0.25
            position = 1
        elif breakout_down and downtrend and vol_confirm and position >= 0:
            signals[i] = -0.25
            position = -1
        elif position == 1 and exit_long:
            signals[i] = 0.0
            position = 0
        elif position == -1 and exit_short:
            signals[i] = 0.0
            position = 0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals