#!/usr/bin/env python3
name = "12h_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # 12h price range for Camarilla levels
    lookback = 1
    prev_high = np.roll(high, lookback)
    prev_low = np.roll(low, lookback)
    prev_close = np.roll(close, lookback)
    
    # Calculate Camarilla levels
    range_ = prev_high - prev_low
    R1 = prev_close + (range_ * 1.1 / 12)
    S1 = prev_close - (range_ * 1.1 / 12)
    R3 = prev_close + (range_ * 1.1 / 4)
    S3 = prev_close - (range_ * 1.1 / 4)
    
    # 1d trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    ema_20_1d = pd.Series(df_1d['close']).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_20_1d)
    daily_uptrend = close > ema_20_1d_aligned
    
    # Volume filter
    volume_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * volume_ma20
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = max(20, 1)  # ensure volume MA and previous bar data ready
    
    for i in range(start_idx, n):
        if np.isnan(daily_uptrend[i]) or np.isnan(volume_ma20[i]) or np.isnan(R1[i]) or np.isnan(S1[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 with daily uptrend and volume
            if close[i] > R1[i] and daily_uptrend[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 with daily downtrend and volume
            elif close[i] < S1[i] and not daily_uptrend[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price breaks below S1 or daily trend flips
            if close[i] < S1[i] or not daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price breaks above R1 or daily trend flips
            if close[i] > R1[i] or daily_uptrend[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals