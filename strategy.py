#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot breakout with 4h EMA34 trend filter and volume confirmation
# Uses 1h Camarilla R3/S3 levels from prior completed 1h bar for structure
# 4h EMA34 provides higher timeframe trend filter to reduce whipsaw in ranging markets
# Volume confirmation (>1.8x 20 EMA) ensures breakout has strong participation
# Session filter (08-20 UTC) reduces noise trades during low liquidity periods
# Discrete sizing 0.20 limits risk and reduces fee churn
# Target: 60-150 total trades over 4 years = 15-37/year for 1h
# Camarilla pivots work well in both bull and bear markets when combined with volume and trend filters

name = "1h_Camarilla_R3S3_4hEMA34_VolumeSpike_Session"
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex, .hour works directly
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 1h data for Camarilla pivot calculation
    df_1h = get_htf_data(prices, '1h')
    if len(df_1h) < 2:
        return np.zeros(n)
    
    # Calculate Camarilla pivot points (R3, S3) from prior completed 1h bar
    high_1h = df_1h['high'].values
    low_1h = df_1h['low'].values
    close_1h = df_1h['close'].values
    
    # Use prior completed 1h bar (shift by 1)
    high_1h_shifted = np.roll(high_1h, 1)
    low_1h_shifted = np.roll(low_1h, 1)
    close_1h_shifted = np.roll(close_1h, 1)
    high_1h_shifted[0] = np.nan
    low_1h_shifted[0] = np.nan
    close_1h_shifted[0] = np.nan
    
    # Calculate pivot point (PP)
    pp = (high_1h_shifted + low_1h_shifted + close_1h_shifted) / 3.0
    
    # Calculate Camarilla levels
    # R3 = PP + (High - Low) * 1.1/4
    # S3 = PP - (High - Low) * 1.1/4
    r3 = pp + ((high_1h_shifted - low_1h_shifted) * 1.1 / 4.0)
    s3 = pp - ((high_1h_shifted - low_1h_shifted) * 1.1 / 4.0)
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1h, s3)
    
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
        if (np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or 
            np.isnan(ema_34_4h_aligned[i]) or np.isnan(vol_ema_20[i]) or
            not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 + price above 4h EMA34 + volume spike
            if close[i] > r3_aligned[i] and close[i] > ema_34_4h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = 0.20
                position = 1
            # Short conditions: price breaks below S3 + price below 4h EMA34 + volume spike
            elif close[i] < s3_aligned[i] and close[i] < ema_34_4h_aligned[i] and volume[i] > (1.8 * vol_ema_20[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price returns to Camarilla midpoint (R3/S3 average) OR price crosses below 4h EMA34
            camarilla_mid = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if not np.isnan(camarilla_mid) and (close[i] < camarilla_mid or close[i] < ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price returns to Camarilla midpoint OR price crosses above 4h EMA34
            camarilla_mid = (r3_aligned[i] + s3_aligned[i]) / 2.0
            if not np.isnan(camarilla_mid) and (close[i] > camarilla_mid or close[i] > ema_34_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals