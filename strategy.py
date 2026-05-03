#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla pivot (R3/S3) breakout with 4h EMA50 trend filter and volume confirmation.
# Long when price breaks above R3 AND close > 4h EMA50 AND volume > 1.5x 20-period MA.
# Short when price breaks below S3 AND close < 4h EMA50 AND volume > 1.5x 20-period MA.
# Exit when price reverts to the pivot point (PP) or 4h EMA50 filter fails.
# Uses 4h/1d for signal direction via EMA50 trend, 1h only for entry timing precision.
# Session filter (08-20 UTC) to reduce noise trades. Target size: 0.20.
# Designed for 60-150 total trades over 4 years (15-37/year) with tight entry conditions.

name = "1h_Camarilla_R3S3_Breakout_4hEMA50_VolumeSpike_Session"
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
    open_time = prices['open_time']
    
    # Session filter: 08-20 UTC (pre-compute to avoid datetime64 issues)
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
    
    # Calculate 1h Camarilla pivots (using previous bar's HLC)
    # Camarilla levels: PP = (H+L+C)/3, R3 = PP + (H-L)*1.1/2, S3 = PP - (H-L)*1.1/2
    # We use previous bar's data to avoid look-ahead
    pp = (np.roll(high, 1) + np.roll(low, 1) + np.roll(close, 1)) / 3
    r3 = pp + (np.roll(high, 1) - np.roll(low, 1)) * 1.1 / 2
    s3 = pp - (np.roll(high, 1) - np.roll(low, 1)) * 1.1 / 2
    # Set first bar to NaN (no previous bar)
    pp[0] = np.nan
    r3[0] = np.nan
    s3[0] = np.nan
    
    # Calculate 1h volume 20-period MA for spike detection
    volume_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(30, n):  # Start after warmup for volume MA
        # Skip if any value is NaN or outside session
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(pp[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or 
            np.isnan(volume_ma_20[i]) or not in_session[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
            
        # Volume spike condition: current 1h volume > 1.5x 20-period volume MA
        volume_spike = volume[i] > (volume_ma_20[i] * 1.5)
        
        if position == 0:
            # Long: price breaks above R3 AND close > 4h EMA50 AND volume spike AND session
            if close[i] > r3[i] and close[i] > ema_50_4h_aligned[i] and volume_spike:
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S3 AND close < 4h EMA50 AND volume spike AND session
            elif close[i] < s3[i] and close[i] < ema_50_4h_aligned[i] and volume_spike:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price reverts to PP OR close < 4h EMA50 (trend fails)
            if close[i] <= pp[i] or close[i] < ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price reverts to PP OR close > 4h EMA50 (trend fails)
            if close[i] >= pp[i] or close[i] > ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals