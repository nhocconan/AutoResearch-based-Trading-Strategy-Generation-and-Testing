# 6H_Camarilla_R3S3_1DTrend_VolumeBreakout
# Hypothesis: 6-hour Camarilla R3/S3 breakout with 1-day trend filter (price > EMA20) and volume spike confirmation.
# Uses daily trend to avoid counter-trend trades, Camarilla levels for structure, volume for momentum.
# Targets 15-35 trades/year to minimize fee drift in 6h timeframe.

name = "6H_Camarilla_R3S3_1DTrend_VolumeBreakout"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:  # Need enough data for EMA20
        return np.zeros(n)
    
    # Calculate daily EMA20 for trend filter
    ema_20_1d = pd.Series(df_1d['close'].values).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    
    # Calculate Camarilla levels from previous day's OHLC
    prev_close = df_1d['close'].shift(1).values
    prev_high = df_1d['high'].shift(1).values
    prev_low = df_1d['low'].shift(1).values
    
    # Camarilla R3, S3, R4, S4 levels
    camarilla_r3 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 12
    camarilla_s3 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 12
    camarilla_r4 = prev_close + 1.1 * (prev_high - prev_low) * 1.1 / 6
    camarilla_s4 = prev_close - 1.1 * (prev_high - prev_low) * 1.1 / 6
    
    # Align Camarilla levels to 6h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume filter: current volume > 2.0x average volume (20-period)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need EMA and volume MA data
    
    for i in range(start_idx, n):
        # Skip if any critical value is NaN
        if (np.isnan(ema_20_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(r4_aligned[i]) or 
            np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i]) or vol_ma[i] == 0):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Volume filter: spike confirmation (2.0x average volume)
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 0:
            # Long: Price breaks above R3 with daily uptrend + volume spike
            if (close[i] > r3_aligned[i] and 
                close[i] > ema_20_1d_aligned[i] and   # Daily uptrend filter
                volume_filter):
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below S3 with daily downtrend + volume spike
            elif (close[i] < s3_aligned[i] and 
                  close[i] < ema_20_1d_aligned[i] and   # Daily downtrend filter
                  volume_filter):
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: Price reaches opposite Camarilla level (R4 for long, S4 for short)
            if position == 1 and close[i] >= r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            elif position == -1 and close[i] <= s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                # Maintain position
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals