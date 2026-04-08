#!/usr/bin/env python3
"""
6h_1w_1d_volume_acceleration_v1
Hypothesis: Volume acceleration + price breakout on 6h with 1w trend filter. 
- Volume surge detection: current volume > 2x 20-period average AND rising for 2 consecutive periods
- Price breakout: close breaks above/below Donchian(20) channel
- Trend filter: 1w EMA(50) direction (bullish if close > EMA50, bearish if close < EMA50)
- Entry: Long on bullish breakout in bullish trend; Short on bearish breakout in bearish trend
- Exit: Opposite breakout or trend reversal
- Position sizing: 0.25 long, -0.25 short
- Designed to capture momentum bursts in both bull and bear markets with volume confirmation
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_1w_1d_volume_acceleration_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 50:
        return np.zeros(n)
    
    # 1w EMA(50) for trend
    close_1w = df_1w['close'].values
    ema_50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_1w_up = close_1w > ema_50_1w
    trend_1w_down = close_1w < ema_50_1w
    
    # Forward fill trend
    trend_1w_up_series = pd.Series(trend_1w_up)
    trend_1w_down_series = pd.Series(trend_1w_down)
    trend_1w_up_ffilled = trend_1w_up_series.ffill().values
    trend_1w_down_ffilled = trend_1w_down_series.ffill().values
    
    # Align 1w trend to 6h
    trend_1w_up_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_up_ffilled)
    trend_1w_down_aligned = align_htf_to_ltf(prices, df_1w, trend_1w_down_ffilled)
    
    # Donchian(20) channels
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().values
    
    # Volume acceleration: current volume > 2x 20-period MA AND rising for 2 periods
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_surge = volume > (2.0 * vol_ma)
    volume_rising = volume > np.roll(volume, 1)  # volume > previous period
    volume_rising_2 = volume_rising & np.roll(volume_rising, 1)  # rising for 2 consecutive periods
    volume_acceleration = volume_surge & volume_rising_2
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start after warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if data not available
        if (np.isnan(trend_1w_up_aligned[i]) or np.isnan(trend_1w_down_aligned[i]) or
            np.isnan(high_20[i]) or np.isnan(low_20[i]) or
            np.isnan(volume_acceleration[i])):
            if position != 0:
                # Hold position until exit conditions met
                pass
            else:
                signals[i] = 0.0
            continue
        
        if position == 1:  # Long position
            # Exit: bearish breakout OR 1w trend turns down
            if (close[i] < low_20[i]) or trend_1w_down_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25  # Position size
                
        elif position == -1:  # Short position
            # Exit: bullish breakout OR 1w trend turns up
            if (close[i] > high_20[i]) or trend_1w_up_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25  # Position size
        else:  # Flat, look for entry
            # Long entry: bullish breakout + 1w uptrend + volume acceleration
            if (close[i] > high_20[i]) and trend_1w_up_aligned[i] and volume_acceleration[i]:
                position = 1
                signals[i] = 0.25
            # Short entry: bearish breakout + 1w downtrend + volume acceleration
            elif (close[i] < low_20[i]) and trend_1w_down_aligned[i] and volume_acceleration[i]:
                position = -1
                signals[i] = -0.25
    
    return signals