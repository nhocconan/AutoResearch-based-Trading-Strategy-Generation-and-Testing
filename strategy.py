#!/usr/bin/env python3
name = "1h_Camarilla_R3_S3_Breakout_4hEMA20_Trend_Volume"
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
    
    # Calculate 4h EMA20 for trend filter (HTF) - ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    ema20_4h = pd.Series(df_4h['close']).ewm(span=20, min_periods=20, adjust=False).mean().values
    ema20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema20_4h)
    
    # Calculate daily high/low/close for Camarilla levels
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Camarilla levels: R3, S3
    hl_range = high_4h - low_4h
    r3 = close_4h + hl_range * 1.25
    s3 = close_4h - hl_range * 1.25
    
    # Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_4h, r3)
    s3_aligned = align_htf_to_ltf(prices, df_4h, s3)
    
    # Volume filter: 20-period EMA for higher threshold
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    # Session filter: 08:00-20:00 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    # Fixed position size to avoid churn
    position_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema20_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema4h = close[i] > ema20_4h_aligned[i]
        price_below_ema4h = close[i] < ema20_4h_aligned[i]
        breakout_long = close[i] > r3_aligned[i]
        breakout_short = close[i] < s3_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 + above 4h EMA20 + volume + session
            if breakout_long and price_above_ema4h and volume_ok[i] and session_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below S3 + below 4h EMA20 + volume + session
            elif breakout_short and price_below_ema4h and volume_ok[i] and session_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions
            if position == 1:
                # Exit: Price crosses below S3 OR trend reverses OR session ends
                if close[i] < s3_aligned[i] or close[i] < ema20_4h_aligned[i] or not session_ok[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above R3 OR trend reverses OR session ends
                if close[i] > r3_aligned[i] or close[i] > ema20_4h_aligned[i] or not session_ok[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals