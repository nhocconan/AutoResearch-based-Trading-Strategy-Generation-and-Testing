#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h strategy using 4h Camarilla R3/S3 breakout with 1d EMA50 trend filter and volume confirmation
# Uses 4h for signal direction (breakouts from Camarilla levels) and 1d for trend/volume filters to reduce noise
# Session filter (08-20 UTC) further reduces false signals. Designed for 15-37 trades/year on 1h timeframe
# to minimize fee drag while capturing high-probability breakouts in both bull and bear markets.

name = "1h_Camarilla_R3S3_Breakout_1dEMA50_VolumeSpike"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for Camarilla pivots (using previous 4h bar's OHLC)
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 30:
        return np.zeros(n)
    
    # Calculate 4h EMA50 for trend filter
    ema_50_4h = pd.Series(df_4h['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate 4h volume spike (volume > 2.0 * 20-period EMA of volume)
    vol_ema_20_4h = pd.Series(df_4h['volume'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike_4h = df_4h['volume'].values > (2.0 * vol_ema_20_4h)
    
    # Get 1d data for additional trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # Calculate 1d EMA50 for trend filter
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align 4h indicators to 1h timeframe
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    volume_spike_4h_aligned = align_htf_to_ltf(prices, df_4h, volume_spike_4h)
    
    # Align 1d EMA50 to 1h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    # Calculate Camarilla levels from previous 4h bar
    prev_close_4h = df_4h['close'].shift(1).values
    prev_high_4h = df_4h['high'].shift(1).values
    prev_low_4h = df_4h['low'].shift(1).values
    
    camarilla_r3_4h = prev_close_4h + ((prev_high_4h - prev_low_4h) * 1.1 / 4)
    camarilla_s3_4h = prev_close_4h - ((prev_high_4h - prev_low_4h) * 1.1 / 4)
    
    # Align Camarilla levels to 1h timeframe
    camarilla_r3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_r3_4h)
    camarilla_s3_4h_aligned = align_htf_to_ltf(prices, df_4h, camarilla_s3_4h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient warmup for indicators
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(volume_spike_4h_aligned[i]) or 
            np.isnan(ema_50_1d_aligned[i]) or np.isnan(camarilla_r3_4h_aligned[i]) or 
            np.isnan(camarilla_s3_4h_aligned[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Determine trend direction: both 4h and 1d EMA50 must agree
        is_uptrend = (close[i] > ema_50_4h_aligned[i]) and (close[i] > ema_50_1d_aligned[i])
        is_downtrend = (close[i] < ema_50_4h_aligned[i]) and (close[i] < ema_50_1d_aligned[i])
        
        if position == 0:
            # Long: price breaks above R3 with volume spike in uptrend
            if (close[i] > camarilla_r3_4h_aligned[i] and 
                close[i-1] <= camarilla_r3_4h_aligned[i-1] and 
                is_uptrend and volume_spike_4h_aligned[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 with volume spike in downtrend
            elif (close[i] < camarilla_s3_4h_aligned[i] and 
                  close[i-1] >= camarilla_s3_4h_aligned[i-1] and 
                  is_downtrend and volume_spike_4h_aligned[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price re-enters below R3 or trend changes
            if (close[i] < camarilla_r3_4h_aligned[i]) or not is_uptrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price re-enters above S3 or trend changes
            if (close[i] > camarilla_s3_4h_aligned[i]) or not is_downtrend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals