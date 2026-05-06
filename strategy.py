#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1h Camarilla R3/S3 breakout with 4h EMA50 trend filter, 1d volume spike confirmation, and session filter (08-20 UTC)
# Uses 4h/1d for signal direction, 1h only for entry timing precision
# Long when price breaks above R3 AND close > 4h EMA50 (uptrend) AND 1d volume > 2.0 * 20-bar avg volume AND session 08-20 UTC
# Short when price breaks below S3 AND close < 4h EMA50 (downtrend) AND 1d volume > 2.0 * 20-bar avg volume AND session 08-20 UTC
# Exit when price reverts to the 4h EMA50 level (mean reversion to trend)
# Discrete sizing 0.20 to control fee drag and drawdown
# Target: 60-150 total trades over 4 years (15-37/year) for 1h timeframe

name = "1h_Camarilla_R3S3_4hEMA50_1dVolumeSpike_Session_v2"
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
    
    # Calculate Camarilla levels (based on previous bar's range)
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    prev_close = np.roll(close, 1)
    prev_high[0] = high[0]
    prev_low[0] = low[0]
    prev_close[0] = close[0]
    
    camarilla_range = prev_high - prev_low
    r3 = prev_close + 1.1 * camarilla_range / 2.0
    s3 = prev_close - 1.1 * camarilla_range / 2.0
    
    # Get 4h data ONCE before loop for EMA50 trend filter
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    
    # Calculate 4h EMA50
    close_4h_series = pd.Series(close_4h)
    ema_50_4h = close_4h_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align HTF indicators to 1h timeframe (wait for completed HTF bar)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # Get 1d data ONCE before loop for volume confirmation
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    volume_1d = df_1d['volume'].values
    
    # Calculate 1d volume confirmation: volume > 2.0 * 20-bar average volume
    avg_volume_20_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_spike_1d = volume_1d > (2.0 * avg_volume_20_1d)
    
    # Align 1d volume spike to 1h timeframe
    volume_spike_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_spike_1d.astype(float))
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(open_time).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after warmup period
        # Skip if any value is NaN
        if (np.isnan(r3[i]) or np.isnan(s3[i]) or np.isnan(ema_50_4h_aligned[i]) or 
            np.isnan(volume_spike_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Camarilla breakout signals with trend, volume, and session filters
            # Long: break above R3 AND uptrend AND volume spike AND session
            if close[i] > r3[i] and close[i] > ema_50_4h_aligned[i] and volume_spike_1d_aligned[i] and in_session[i]:
                signals[i] = 0.20
                position = 1
            # Short: break below S3 AND downtrend AND volume spike AND session
            elif close[i] < s3[i] and close[i] < ema_50_4h_aligned[i] and volume_spike_1d_aligned[i] and in_session[i]:
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Exit long: price reverts to 4h EMA50 (mean reversion to trend)
            if close[i] <= ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Exit short: price reverts to 4h EMA50 (mean reversion to trend)
            if close[i] >= ema_50_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals