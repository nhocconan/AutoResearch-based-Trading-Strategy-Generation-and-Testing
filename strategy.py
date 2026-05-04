#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA34 trend filter and volume confirmation
# Uses Camarilla pivot levels from prior completed 1d for structure, 4h EMA34 for trend filter
# Volume confirmation (>1.5x 20 EMA) ensures breakout has strong participation
# Session filter (08-20 UTC) reduces noise trades during low-liquidity periods
# Discrete sizing 0.20 limits risk and reduces fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# 4h EMA34 ensures we only trade with the higher timeframe trend, reducing whipsaw in both bull and bear markets.

name = "1h_Camarilla_R3S3_4hEMA34_VolumeConfirm_Session"
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
    
    # Pre-compute session hours (08-20 UTC) to avoid datetime operations in loop
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1d data for Camarilla pivot calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels using prior completed 1d bar
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla R3, S3, R4, S4 levels
    # Camarilla: R4 = close + 1.1*(high-low)*1.1/2, R3 = close + 1.1*(high-low)*1.1/4
    #          S3 = close - 1.1*(high-low)*1.1/4, S4 = close - 1.1*(high-low)*1.1/2
    range_1d = high_1d - low_1d
    camarilla_r3 = close_1d + 1.1 * range_1d * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * range_1d * 1.1 / 4
    camarilla_r4 = close_1d + 1.1 * range_1d * 1.1 / 2
    camarilla_s4 = close_1d - 1.1 * range_1d * 1.1 / 2
    
    # Shift by 1 to use only completed 1d bar (avoid look-ahead)
    camarilla_r3_shifted = np.roll(camarilla_r3, 1)
    camarilla_s3_shifted = np.roll(camarilla_s3, 1)
    camarilla_r4_shifted = np.roll(camarilla_r4, 1)
    camarilla_s4_shifted = np.roll(camarilla_s4, 1)
    camarilla_r3_shifted[0] = np.nan
    camarilla_s3_shifted[0] = np.nan
    camarilla_r4_shifted[0] = np.nan
    camarilla_s4_shifted[0] = np.nan
    
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3_shifted)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3_shifted)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4_shifted)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4_shifted)
    
    # Get 4h data for EMA34 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 34:
        return np.zeros(n)
    
    # Calculate 4h EMA34 trend filter
    close_4h = df_4h['close'].values
    ema_34_4h = pd.Series(close_4h).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_34_4h)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ema_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + price above 4h EMA34 + volume spike
            if close[i] > camarilla_r3_aligned[i] and close[i] > ema_34_4h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 + price below 4h EMA34 + volume spike
            elif close[i] < camarilla_s3_aligned[i] and close[i] < ema_34_4h_aligned[i] and volume[i] > (1.5 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla midpoint (R3/S3) OR price crosses below 4h EMA34
            camarilla_mid = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2.0
            if not np.isnan(camarilla_mid) and (close[i] < camarilla_mid or close[i] < ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla midpoint (R3/S3) OR price crosses above 4h EMA34
            camarilla_mid = (camarilla_r3_aligned[i] + camarilla_s3_aligned[i]) / 2.0
            if not np.isnan(camarilla_mid) and (close[i] > camarilla_mid or close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals