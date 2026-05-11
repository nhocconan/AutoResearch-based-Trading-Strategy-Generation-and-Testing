#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Calculate Camarilla levels from previous day
    daily_data = get_htf_data(prices, '1d')
    if len(daily_data) < 2:
        return np.zeros(n)
    
    # Previous day's OHLC
    prev_close = daily_data['close'].iloc[:-1].values
    prev_high = daily_data['high'].iloc[:-1].values
    prev_low = daily_data['low'].iloc[:-1].values
    
    # Calculate Camarilla levels
    range_prev = prev_high - prev_low
    R3 = prev_close + (range_prev * 1.1 / 4)
    S3 = prev_close - (range_prev * 1.1 / 4)
    
    # Align to 4h timeframe
    R3_aligned = align_htf_to_ltf(prices, daily_data, R3)
    S3_aligned = align_htf_to_ltf(prices, daily_data, S3)
    
    # Daily trend filter: price above/below 34 EMA
    daily_close = daily_data['close'].values
    ema_34_daily = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, daily_data, ema_34_daily)
    daily_uptrend = close > ema_34_aligned
    
    # Volume filter: current volume > 1.5x 20-period average
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(34, 20)  # Wait for indicators to be ready
    
    for i in range(start_idx, n):
        if np.isnan(R3_aligned[i]) or np.isnan(S3_aligned[i]) or np.isnan(daily_uptrend[i]) or np.isnan(volume_ma20[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R3, daily uptrend, volume confirmation
            if close[i] > R3_aligned[i] and daily_uptrend[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S3, daily downtrend, volume confirmation
            elif close[i] < S3_aligned[i] and not daily_uptrend[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S3 or daily trend turns down
            if close[i] < S3_aligned[i] or not daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R3 or daily trend turns up
            if close[i] > R3_aligned[i] or daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals