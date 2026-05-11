#!/usr/bin/env python3
name = "1h_Camarilla_R3S3_Breakout_4hTrend_VolumeSpike"
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
    
    # 1. Load 4h data ONCE for trend filter
    df_4h = get_htf_data(prices, '4h')
    
    # 2. 4h EMA34 for trend filter
    ema34_4h = pd.Series(df_4h['close']).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_4h_aligned = align_htf_to_ltf(prices, df_4h, ema34_4h)
    
    # 3. Load 1d data ONCE for Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    
    # 4. Calculate daily high/low/close for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 5. Camarilla levels: R3, S3
    hl_range = high_1d - low_1d
    r3 = close_1d + hl_range * 1.25
    s3 = close_1d - hl_range * 1.25
    
    # 6. Align Camarilla levels to 1h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 7. Volume filter: 20-period EMA for higher threshold
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 2.0
    
    # 8. Session filter: 08-20 UTC
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    session_ok = (hours >= 8) & (hours <= 20)
    
    # 9. Fixed position size to avoid churn
    position_size = 0.20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_4h_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ok[i]) or np.isnan(session_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema4h = close[i] > ema34_4h_aligned[i]
        price_below_ema4h = close[i] < ema34_4h_aligned[i]
        breakout_long = close[i] > r3_aligned[i]
        breakout_short = close[i] < s3_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 + above 4h EMA34 + volume spike + session
            if breakout_long and price_above_ema4h and volume_ok[i] and session_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below S3 + below 4h EMA34 + volume spike + session
            elif breakout_short and price_below_ema4h and volume_ok[i] and session_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - simplified to reduce churn
            if position == 1:
                # Exit: Price crosses below S3 OR trend reverses
                if close[i] < s3_aligned[i] or close[i] < ema34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above R3 OR trend reverses
                if close[i] > r3_aligned[i] or close[i] > ema34_4h_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals