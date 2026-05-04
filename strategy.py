#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume spike confirmation
# Uses 4h for signal direction (Camarilla breakouts), 1h only for entry timing precision
# 1d EMA50 provides strong trend filter to avoid counter-trend trades in bear markets
# Volume confirmation (>2.0x 20 EMA) ensures breakout has strong participation
# Session filter (08-20 UTC) reduces noise during low-liquidity periods
# Discrete sizing 0.20 limits risk and reduces fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h.
# 4h Camarilla levels from prior completed 4h bar to avoid look-ahead
# 1d EMA50 uses prior completed 1d bar for trend alignment

name = "1h_Camarilla_R3S3_1dEMA50_VolumeSpike_Session"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Pre-compute session hours once (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 4h data for Camarilla pivot levels (prior completed 4h bar)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla levels (R3, S3) from prior completed 4h bar
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Shift by 1 to use prior completed 4h bar
    high_4h_shifted = np.roll(high_4h, 1)
    low_4h_shifted = np.roll(low_4h, 1)
    close_4h_shifted = np.roll(close_4h, 1)
    high_4h_shifted[0] = np.nan
    low_4h_shifted[0] = np.nan
    close_4h_shifted[0] = np.nan
    
    camarilla_range = high_4h_shifted - low_4h_shifted
    r3 = close_4h_shifted + 1.1 * camarilla_range * 1.1 / 4
    s3 = close_4h_shifted - 1.1 * camarilla_range * 1.1 / 4
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Get 1d data for EMA50 trend filter (prior completed 1d bar)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50 trend filter
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Volume confirmation: 20-period EMA of volume on 1h timeframe
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(vol_ema_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above Camarilla R3 + price above 1d EMA50 + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below Camarilla S3 + price below 1d EMA50 + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_50_1d_aligned[i] and volume[i] > (2.0 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla midpoint OR price crosses below 1d EMA50
            camarilla_mid = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if not np.isnan(camarilla_mid) and (close[i] < camarilla_mid or close[i] < ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla midpoint OR price crosses above 1d EMA50
            camarilla_mid = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if not np.isnan(camarilla_mid) and (close[i] > camarilla_mid or close[i] > ema_50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals