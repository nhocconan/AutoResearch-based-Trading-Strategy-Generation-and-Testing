#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation
# Uses Camarilla pivot levels (R3, S3) from 4h OHLC for institutional breakout zones
# 4h EMA50 ensures alignment with intermediate-term trend to avoid counter-trend trades
# Volume spike (2.0x 20-bar MA) confirms institutional participation
# Session filter (08-20 UTC) reduces noise trades
# Designed for 60-150 total trades over 4 years (15-37/year) on 1h timeframe
# Works in bull markets (breakouts with trend) and bear markets (mean reversion at extremes)

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC) - open_time is already datetime64[ms]
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Calculate 4h EMA50 for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Volume confirmation: 2.0x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (2.0 * vol_ma)
    
    # Calculate 4h Camarilla levels (R3, S3)
    df_4h_cam = get_htf_data(prices, '4h')
    if len(df_4h_cam) < 2:
        return np.zeros(n)
    
    high_4h = df_4h_cam['high'].values
    low_4h = df_4h_cam['low'].values
    close_4h_cam = df_4h_cam['close'].values
    
    # Camarilla: R3 = close + 1.1*(high-low)/4, S3 = close - 1.1*(high-low)/4
    camarilla_r3 = close_4h_cam + 1.1 * (high_4h - low_4h) / 4
    camarilla_s3 = close_4h_cam - 1.1 * (high_4h - low_4h) / 4
    
    # Align to 1h timeframe (wait for 4h close)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h_cam, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h_cam, camarilla_s3)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup (need enough data for EMA50 and volume MA)
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if not in trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Check for NaN values in indicators
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ma[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:  # Flat - look for new entries
            # Long entry: Price breaks above R3 AND price > 4h EMA50 (bullish trend) AND volume spike
            if (close[i] > camarilla_r3_aligned[i] and 
                close[i] > ema_50_4h_aligned[i] and 
                volume_spike[i]):
                signals[i] = 0.20
                position = 1
            # Short entry: Price breaks below S3 AND price < 4h EMA50 (bearish trend) AND volume spike
            elif (close[i] < camarilla_s3_aligned[i] and 
                  close[i] < ema_50_4h_aligned[i] and 
                  volume_spike[i]):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        
        elif position == 1:  # Long position
            # Exit: Price breaks below S3 (reversion to mean) OR price below 4h EMA50 (trend change)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        
        elif position == -1:  # Short position
            # Exit: Price breaks above R3 (reversion to mean) OR price above 4h EMA50 (trend change)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals