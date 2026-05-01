#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla R4/S4 breakout with 12h EMA34 trend filter and volume confirmation
# Uses 12h EMA for intermediate trend direction (more responsive than weekly, filters noise)
# Camarilla R4/S4 levels from 1d provide strong intraday support/resistance
# Volume spike confirms breakout authenticity
# Designed for low frequency (75-200 trades over 4 years) to minimize fee drag
# Works in bull/bear via trend filter + breakout logic + proper position sizing

name = "4h_Camarilla_R4S4_Breakout_12hEMA34_Trend_VolumeSpike_v1"
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
    
    # 1d data for Camarilla pivot calculation (yesterday's OHLC)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # 12h HTF data for EMA34 trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    # 12h EMA34 for trend filter
    close_12h = df_12h['close'].values
    ema_34_12h = pd.Series(close_12h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_34_12h)
    
    # Calculate Camarilla levels from previous 1d bar
    # Camarilla: PP = (H+L+C)/3, Range = H-L
    # R4 = C + (H-L)*1.1, S4 = C - (H-L)*1.1
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Shift by 1 to use previous day's data (no look-ahead)
    high_1d_shifted = np.concatenate([[np.nan], high_1d[:-1]])
    low_1d_shifted = np.concatenate([[np.nan], low_1d[:-1]])
    close_1d_shifted = np.concatenate([[np.nan], close_1d[:-1]])
    
    pivot_point = (high_1d_shifted + low_1d_shifted + close_1d_shifted) / 3.0
    daily_range = high_1d_shifted - low_1d_shifted
    
    # Camarilla R4 and S4 levels (more extreme than R3/S3)
    r4_level = close_1d_shifted + (daily_range * 1.1)
    s4_level = close_1d_shifted - (daily_range * 1.1)
    
    # Align Camarilla levels to 4h timeframe
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4_level)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4_level)
    
    # Volume confirmation: current volume > 2.0 * 20-period average volume
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (volume_ma_20 * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup for all indicators
    start_idx = max(34, 20)  # Need 12h EMA34 and volume MA20
    
    for i in range(start_idx, n):
        if (np.isnan(ema_34_12h_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(volume_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Breakout conditions using Camarilla levels
        breakout_up = close[i] > r4_aligned[i-1]  # Break above R4
        breakout_down = close[i] < s4_aligned[i-1]  # Break below S4
        
        # Trend filter: price above/below 12h EMA34
        uptrend = close[i] > ema_34_12h_aligned[i]
        downtrend = close[i] < ema_34_12h_aligned[i]
        
        # Volume confirmation
        vol_spike