#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with volume confirmation and 4h/1d trend filter
# Uses 4h EMA50 and 1d EMA50 for trend alignment to avoid counter-trend trades
# Uses 1h volume > 1.8x 20-period EMA for breakout confirmation
# Designed for 1h timeframe targeting 15-35 trades/year with discrete sizing (0.20)
# Session filter (08-20 UTC) reduces noise during low-volume periods
# Works in bull markets (breakouts with volume in uptrend) and bear markets (breakouts with volume in downtrend)

name = "1h_Camarilla_R3S3_VolumeSpike_4h1dEMA50_Trend"
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
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    
    # Calculate 4h EMA50
    close_4h = df_4h['close'].values
    ema_50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Calculate 1d EMA50
    close_1d = df_1d['close'].values
    ema_50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate 1h Camarilla levels (based on previous day's range)
    # We need daily high/low/close for Camarilla calculation
    df_1d_for_camarilla = get_htf_data(prices, '1d')
    if len(df_1d_for_camarilla) < 2:
        return np.zeros(n)
    
    # Get previous day's OHLC for Camarilla levels
    high_1d = df_1d_for_camarilla['high'].values
    low_1d = df_1d_for_camarilla['low'].values
    close_1d = df_1d_for_camarilla['close'].values
    
    # Calculate Camarilla levels: R3, R4, S3, S4
    # R4 = Close + 1.1*(High-Low)*1.1/2
    # R3 = Close + 1.1*(High-Low)*1.1/4
    # S3 = Close - 1.1*(High-Low)*1.1/4
    # S4 = Close - 1.1*(High-Low)*1.1/2
    range_1d = high_1d - low_1d
    camarilla_r4 = close_1d + 1.1 * range_1d * 1.1 / 2
    camarilla_r3 = close_1d + 1.1 * range_1d * 1.1 / 4
    camarilla_s3 = close_1d - 1.1 * range_1d * 1.1 / 4
    camarilla_s4 = close_1d - 1.1 * range_1d * 1.1 / 2
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_r3)
    camarilla_r4_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_r4)
    camarilla_s3_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_s3)
    camarilla_s4_aligned = align_htf_to_ltf(prices, df_1d_for_camarilla, camarilla_s4)
    
    # Calculate 1h volume EMA(20) for volume confirmation
    vol_ema_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(ema_50_1d_aligned[i]) or 
            np.isnan(camarilla_r3_aligned[i]) or np.isnan(camarilla_s3_aligned[i]) or
            np.isnan(vol_ema_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume confirmation: current 1h volume > 1.8 x 20-period EMA
        volume_confirmed = volume[i] > (1.8 * vol_ema_20[i])
        
        # Trend filter: both 4h and 1d EMA50 must agree on direction
        uptrend = (close[i] > ema_50_4h_aligned[i]) and (close[i] > ema_50_1d_aligned[i])
        downtrend = (close[i] < ema_50_4h_aligned[i]) and (close[i] < ema_50_1d_aligned[i])
        
        if position == 0:
            # Long: price breaks above Camarilla R3 + volume confirmation + uptrend
            if (close[i] > camarilla_r3_aligned[i] and volume_confirmed and uptrend):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below Camarilla S3 + volume confirmation + downtrend
            elif (close[i] < camarilla_s3_aligned[i] and volume_confirmed and downtrend):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price falls below Camarilla S3 OR trend changes to downtrend
            if (close[i] < camarilla_s3_aligned[i]) or (not uptrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price rises above Camarilla R3 OR trend changes to uptrend
            if (close[i] > camarilla_r3_aligned[i]) or (not downtrend):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals