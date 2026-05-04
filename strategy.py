#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA(50) trend filter and volume spike confirmation
# Uses 4h Camarilla pivot levels for institutional breakout timing within the 1h chart
# 4h EMA50 ensures alignment with higher timeframe trend to avoid counter-trend whipsaws
# Volume confirmation (>1.8x 20 EMA volume) filters false breakouts
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Discrete sizing 0.20 targets 60-150 total trades over 4 years (15-37/year)
# Works in bull markets (continuation at R3/R4) and bear markets (continuation at S3/S4)
# Focus on BTC/ETH by requiring 4h trend alignment (avoids SOL-only bias)

name = "1h_Camarilla_R3S3_4hEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    open_time = prices['open_time'].values
    
    # Precompute session hours (08-20 UTC) once before loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla calculation and EMA trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:  # Need enough data for EMA50 calculation
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    volume_4h = df_4h['volume'].values
    
    # Calculate prior completed 4h Camarilla levels (R3, R3, S3, S4)
    # Camarilla: Range = high - low
    # R4 = close + Range * 1.1/2
    # R3 = close + Range * 1.1/4
    # S3 = close - Range * 1.1/4
    # S4 = close - Range * 1.1/2
    daily_range_4h = high_4h - low_4h
    camarilla_r4_4h = close_4h + daily_range_4h * 1.1 / 2
    camarilla_r3_4h = close_4h + daily_range_4h * 1.1 / 4
    camarilla_s3_4h = close_4h - daily_range_4h * 1.1 / 4
    camarilla_s4_4h = close_4h - daily_range_4h * 1.1 / 2
    
    # Shift to use prior completed 4h bar (avoid look-ahead)
    camarilla_r4_4h_shifted = np.roll(camarilla_r4_4h, 1)
    camarilla_r3_4h_shifted = np.roll(camarilla_r3_4h, 1)
    camarilla_s3_4h_shifted = np.roll(camarilla_s3_4h, 1)
    camarilla_s4_4h_shifted = np.roll(camarilla_s4_4h, 1)
    camarilla_r4_4h_shifted[0] = np.nan
    camarilla_r3_4h_shifted[0] = np.nan
    camarilla_s3_4h_shifted[0] = np.nan
    camarilla_s4_4h_shifted[0] = np.nan
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r4_4h_shifted)
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h_shifted)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h_shifted)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s4_4h_shifted)
    
    # Calculate 4h EMA(50) trend filter from prior completed 4h bar
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_shifted = np.roll(ema_50_4h, 1)
    ema_50_4h_shifted[0] = np.nan
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h_shifted)
    
    # Volume confirmation: 20-period EMA of volume (1h)
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_50_4h_aligned[i]) or np.isnan(vol_ema_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 AND price > 4h EMA50 AND volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_50_4h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 AND price < 4h EMA50 AND volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_50_4h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla S3 OR price crosses below 4h EMA50
            if close[i] < camarilla_s3_aligned[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla R3 OR price crosses above 4h EMA50
            if close[i] > camarilla_r3_aligned[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals