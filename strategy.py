#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter and volume confirmation + session filter (08-20 UTC)
# Uses 4h EMA50 for trend direction and Camarilla levels from 1h for entry/exit
# Volume confirmation requires 2.0x average volume to ensure strong participation
# Session filter reduces noise trades during low-liquidity hours
# Target: 60-150 total trades over 4 years = 15-37/year for 1h
# Uses discrete position sizing (0.20) to minimize fee churn
# Works in both bull and bear markets by following the 4h trend direction and using Camarilla for structure

name = "1h_Camarilla_R3_S3_Breakout_4hEMA50_Trend_Volume_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Calculate Camarilla levels from previous 1h bar (HLC of completed 1h bar)
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Calculate Camarilla levels using previous completed 1h bar
    camarilla_r3 = close_1h + (high_1h - low_1h) * 1.1 / 4
    camarilla_s3 = close_1h - (high_1h - low_1h) * 1.1 / 4
    camarilla_r4 = close_1h + (high_1h - low_1h) * 1.1 / 2
    camarilla_s4 = close_1h - (high_1h - low_1h) * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe (use previous hour's levels)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r3)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1h, camarilla_r4)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1h, camarilla_s4)
    
    # Volume confirmation: 20-period EMA on 1h volume
    vol_series = pd.Series(volume)
    vol_ema_20 = vol_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start from 100 to have valid indicators
        # Skip if any value is NaN
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(camarilla_r3_aligned[i]) or 
            np.isnan(camarilla_s3_aligned[i]) or np.isnan(camarilla_r4_aligned[i]) or 
            np.isnan(camarilla_s4_aligned[i]) or np.isnan(vol_ema_20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Session filter: only trade during 08-20 UTC
        if not in_session[i]:
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume spike: current volume > 2.0 x 20-period EMA (tight to avoid overtrading)
        volume_spike = volume[i] > (2.0 * vol_ema_20[i])
        
        # Camarilla R3/S3 breakout with 4h trend filter
        # Long: Price breaks above R3 + volume spike + price above 4h EMA50 (uptrend)
        # Short: Price breaks below S3 + volume spike + price below 4h EMA50 (downtrend)
        if position == 0:
            if (close[i] > camarilla_r3_aligned[i] and volume_spike and 
                close[i] > ema_50_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            elif (close[i] < camarilla_s3_aligned[i] and volume_spike and 
                  close[i] < ema_50_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: Price breaks below S3 (reversal) OR price below 4h EMA50 (trend change)
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: Price breaks above R3 (reversal) OR price above 4h EMA50 (trend change)
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals