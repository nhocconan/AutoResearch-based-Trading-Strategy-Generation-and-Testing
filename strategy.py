#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h strategy using Camarilla R3/S3 levels from 1d + volume spike + 1w EMA34 trend filter.
# Enter long when price touches or breaks above Camarilla R3 with volume > 2.0x 20-bar average and price > 1w EMA34.
# Enter short when price touches or breaks below Camarilla S3 with volume > 2.0x 20-bar average and price < 1w EMA34.
# Exit on touch of Camarilla S3 (for longs) or R3 (for shorts) or opposite 1w EMA34 crossover.
# Uses discrete position sizing (0.25) to limit drawdown. Target: 50-150 trades over 4 years.
# Camarilla levels provide institutional support/resistance, volume confirms breakout strength,
# 1w EMA34 filters counter-trend noise in both bull and bear markets.

name = "12h_Camarilla_R3S3_1wEMA34_VolumeSpike_v1"
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
    
    # Get 1d data for Camarilla calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d Camarilla levels (R3, S3)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    camarilla_r3 = np.full_like(close_1d, np.nan)
    camarilla_s3 = np.full_like(close_1d, np.nan)
    
    for i in range(len(close_1d)):
        if i == 0:
            camarilla_r3[i] = np.nan
            camarilla_s3[i] = np.nan
        else:
            # Camarilla formulas using previous day's OHLC
            prev_close = close_1d[i-1]
            prev_high = high_1d[i-1]
            prev_low = low_1d[i-1]
            rang = prev_high - prev_low
            camarilla_r3[i] = prev_close + rang * 1.1 / 4
            camarilla_s3[i] = prev_close - rang * 1.1 / 4
    
    # Align 1d Camarilla levels to 12h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    
    # Get 1w data for EMA34 trend filter (HTF)
    df_1w = get_htf_data(prices, '1w')
    
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # Calculate 1w EMA34
    close_1w = df_1w['close'].values
    ema_34_1w = pd.Series(close_1w).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align 1w EMA34 to 12h timeframe
    ema_34_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_34_1w)
    
    # Calculate 12h volume confirmation: >2.0x 20-bar average volume
    volume_series = pd.Series(volume)
    volume_ma_20 = volume_series.rolling(window=20, min_periods=20).mean().values
    volume_confirm = volume > 2.0 * volume_ma_20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Ensure sufficient history for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_1w_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Camarilla level conditions with volume confirmation and trend filter
        long_entry = close[i] >= camarilla_r3_aligned[i] and volume_confirm[i] and close[i] > ema_34_1w_aligned[i]
        short_entry = close[i] <= camarilla_s3_aligned[i] and volume_confirm[i] and close[i] < ema_34_1w_aligned[i]
        
        # Exit conditions: touch opposite Camarilla level or trend reversal
        long_exit = close[i] <= camarilla_s3_aligned[i] or close[i] < ema_34_1w_aligned[i]
        short_exit = close[i] >= camarilla_r3_aligned[i] or close[i] > ema_34_1w_aligned[i]
        
        # Handle entries and exits
        if long_entry and position <= 0:
            signals[i] = 0.25
            position = 1
        elif short_entry and position >= 0:
            signals[i] = -0.25
            position = -1
        elif (position == 1 and long_exit) or (position == -1 and short_exit):
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