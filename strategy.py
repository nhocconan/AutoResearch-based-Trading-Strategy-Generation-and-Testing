#!/usr/bin/env python3
name = "12h_Camarilla_R3_S3_Breakout_1dEMA34_Trend_Volume"
timeframe = "12h"
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
    
    # 1. Load 1d data ONCE for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    ema34_1d = pd.Series(df_1d['close']).ewm(span=34, min_periods=34, adjust=False).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # 2. Load 1d data ONCE for Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # 3. Camarilla levels: R3, S3 (outer levels for fewer, stronger signals)
    hl_range = high_1d - low_1d
    r3 = close_1d + hl_range * 1.5000
    s3 = close_1d - hl_range * 1.5000
    
    # 4. Align Camarilla levels to 12h timeframe
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 5. Volume filter: 20-period EMA for spike detection
    vol_ema20 = pd.Series(volume).ewm(span=20, min_periods=20, adjust=False).mean().values
    volume_ok = volume > vol_ema20 * 1.5
    
    # 6. Fixed position size to avoid churn
    position_size = 0.25
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start after warmup
    start_idx = 100
    
    for i in range(start_idx, n):
        # Skip if any required data is invalid
        if (np.isnan(ema34_1d_aligned[i]) or np.isnan(r3_aligned[i]) or 
            np.isnan(s3_aligned[i]) or np.isnan(volume_ok[i])):
            if position == 1:
                signals[i] = 0.0
            elif position == -1:
                signals[i] = 0.0
            else:
                signals[i] = 0.0
            continue
        
        # Conditions
        price_above_ema1d = close[i] > ema34_1d_aligned[i]
        price_below_ema1d = close[i] < ema34_1d_aligned[i]
        breakout_long = close[i] > r3_aligned[i]
        breakout_short = close[i] < s3_aligned[i]
        
        if position == 0:
            # Long: Price breaks above R3 + above 1d EMA34 + volume spike
            if breakout_long and price_above_ema1d and volume_ok[i]:
                signals[i] = position_size
                position = 1
            # Short: Price breaks below S3 + below 1d EMA34 + volume spike
            elif breakout_short and price_below_ema1d and volume_ok[i]:
                signals[i] = -position_size
                position = -1
        else:
            # Exit conditions - simplified to reduce churn
            if position == 1:
                # Exit: Price crosses below S3 OR trend reverses
                if close[i] < s3_aligned[i] or close[i] < ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = position_size
            elif position == -1:
                # Exit: Price crosses above R3 OR trend reverses
                if close[i] > r3_aligned[i] or close[i] > ema34_1d_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -position_size
    
    return signals