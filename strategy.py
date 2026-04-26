#!/usr/bin/env python3
"""
1h_Camarilla_R1_S1_Breakout_4hEMA20_Trend_VolumeSpike_v2
Hypothesis: On 1h timeframe, Camarilla R1/S1 breakouts with 4h EMA20 trend filter and volume confirmation (>1.8x avg) provides robust directional signals. Uses 4h/1d for signal direction (trend and Camarilla levels) and 1h only for entry timing. Session filter (08-20 UTC) reduces noise. Discrete sizing (0.0, ±0.20) minimizes fee churn. Targets 60-150 trades over 4 years (15-37/year) for optimal 1h frequency. Weekly trend avoided due to excessive whipsaw in bear markets; 4h EMA20 provides cleaner trend.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Pre-compute session hours for filter (08-20 UTC)
    hours = prices.index.hour  # prices.index is DatetimeIndex
    in_session = (hours >= 8) & (hours <= 20)
    
    # Get 4h data for HTF trend filter and Camarilla calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 20:  # need enough for EMA
        return np.zeros(n)
    
    # Calculate 4h EMA20 for trend filter
    close_4h = pd.Series(df_4h['close'].values)
    ema_20_4h = close_4h.ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Get 1d data for Camarilla levels (previous day's range)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close for Camarilla
    prev_high = df_1d['high'].shift(1).values  # shift(1) for previous completed day
    prev_low = df_1d['low'].shift(1).values
    prev_close = df_1d['close'].shift(1).values
    
    # Align daily data to 1h
    prev_high_1h = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_1h = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_1h = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels
    range_ = prev_high_1h - prev_low_1h
    # Avoid division by zero
    range_ = np.maximum(range_, 1e-10)
    
    # Camarilla R1, R3, S1, S3
    r1 = prev_close_1h + range_ * 1.1 / 12
    r3 = prev_close_1h + range_ * 1.1 / 4
    s1 = prev_close_1h - range_ * 1.1 / 12
    s3 = prev_close_1h - range_ * 1.1 / 4
    
    # Volume ratio (current / 20-period average) for spike confirmation
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    vol_ratio = volume / np.maximum(vol_ma, 1e-10)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup: need previous day data + EMA warmup + volume MA
    start_idx = max(20, 20) + 1
    
    for i in range(start_idx, n):
        # Skip if any data not ready or outside session
        if (np.isnan(ema_20_4h_aligned[i]) or np.isnan(r1[i]) or np.isnan(r3[i]) or
            np.isnan(s1[i]) or np.isnan(s3[i]) or np.isnan(vol_ratio[i]) or
            np.isnan(prev_high_1h[i]) or np.isnan(prev_low_1h[i]) or
            not in_session[i]):
            # Hold current position
            if position == 0:
                signals[i] = 0.0
            elif position == 1:
                signals[i] = 0.20
            else:
                signals[i] = -0.20
            continue
        
        vol_confirmed = vol_ratio[i] > 1.8  # volume at least 1.8x average
        
        if position == 0:
            # Long: price > 4h EMA20 + breaks above R1 + volume
            long_signal = (close[i] > ema_20_4h_aligned[i] and 
                          close[i] > r1[i] and 
                          vol_confirmed)
            
            # Short: price < 4h EMA20 + breaks below S1 + volume
            short_signal = (close[i] < ema_20_4h_aligned[i] and 
                           close[i] < s1[i] and 
                           vol_confirmed)
            
            if long_signal:
                signals[i] = 0.20
                position = 1
            elif short_signal:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Hold long
            signals[i] = 0.20
            # Exit: price closes below 4h EMA20 OR breaks below S1 (reversal)
            if close[i] < ema_20_4h_aligned[i] or close[i] < s1[i]:
                signals[i] = 0.0
                position = 0
        elif position == -1:
            # Hold short
            signals[i] = -0.20
            # Exit: price closes above 4h EMA20 OR breaks above R1 (reversal)
            if close[i] > ema_20_4h_aligned[i] or close[i] > r1[i]:
                signals[i] = 0.0
                position = 0
    
    return signals

name = "1h_Camarilla_R1_S1_Breakout_4hEMA20_Trend_VolumeSpike_v2"
timeframe = "1h"
leverage = 1.0