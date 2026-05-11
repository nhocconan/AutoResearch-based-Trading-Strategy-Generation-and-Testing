#!/usr/bin/env python3
name = "1h_Camarilla_R3_S3_Breakout_4hTrend_Volume"
timeframe = "1h"
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
    
    # 4h trend: close above/below 4h EMA50
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 50:
        return np.zeros(n)
    close_4h = df_4h['close'].values
    ema_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_4h)
    trend_up = close > ema_4h_aligned
    
    # 1h Camarilla levels (based on previous day's range)
    # We'll calculate daily pivot points using 1d data
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Previous day's high, low, close
    prev_high = df_1d['high'].shift(1).values  # previous day's high
    prev_low = df_1d['low'].shift(1).values    # previous day's low
    prev_close = df_1d['close'].shift(1).values # previous day's close
    
    # Align to 1h timeframe
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    prev_close_aligned = align_htf_to_ltf(prices, df_1d, prev_close)
    
    # Calculate Camarilla levels for R3, S3
    # R3 = Close + (High - Low) * 1.1/2
    # S3 = Close - (High - Low) * 1.1/2
    rng = prev_high_aligned - prev_low_aligned
    r3 = prev_close_aligned + rng * 1.1 / 2.0
    s3 = prev_close_aligned - rng * 1.1 / 2.0
    
    # Volume filter: volume > 1.5x 24-period average
    vol_ma24 = pd.Series(volume).rolling(window=24, min_periods=24).mean().values
    volume_filter = volume > 1.5 * vol_ma24
    
    # Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 24  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(ema_4h_aligned[i]) or np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(vol_ma24[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if session_filter[i]:
            if position == 0:
                # Long: price breaks above R3 + 4h uptrend + volume spike
                if close[i] > r3[i] and trend_up[i] and volume_filter[i]:
                    signals[i] = 0.20
                    position = 1
                # Short: price breaks below S3 + 4h downtrend + volume spike
                elif close[i] < s3[i] and not trend_up[i] and volume_filter[i]:
                    signals[i] = -0.20
                    position = -1
            elif position == 1:
                # Long exit: price drops below S3 or 4h trend down
                if close[i] < s3[i] or not trend_up[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.20
            elif position == -1:
                # Short exit: price rises above R3 or 4h trend up
                if close[i] > r3[i] or trend_up[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.20
        else:
            # Outside session: flatten position
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
    
    return signals