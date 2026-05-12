#!/usr/bin/env python3
name = "6h_SlopeMomentum_1dTrend_Filter"
timeframe = "6h"
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
    
    # Daily trend filter: EMA50
    df_1d = get_htf_data(prices, '1d')
    ema50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # 6h price momentum: 3-period slope of close (linear regression slope)
    # Slope = (n*sum(xy) - sum(x)*sum(y)) / (n*sum(x^2) - sum(x)^2) for x=0,1,2
    # Simplified for 3 points: slope = (2*y2 + y1 - y0 - 2*y1) / 2 = (y2 - y0) / 2
    # We'll use close[2] - close[0] for 3-bar slope
    close_series = pd.Series(close)
    slope_3 = (close_series.shift(2) - close_series) / 2  # 3-bar slope: (close[t-2] - close[t]) / 2
    slope_3_values = slope_3.values
    
    # 6h volume filter: volume > 1.5x 20-period average
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: active during major sessions (00-08 Asia, 08-16 London/NY, 16-24 NY)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if daily trend or volume data not ready
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or np.isnan(slope_3_values[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: active during London/NY overlap (08-16 UTC) and Asia (00-08 UTC)
        hour = hours[i]
        in_session = ((0 <= hour <= 8) or (8 <= hour <= 16))
        
        if not in_session:
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long conditions: positive slope (momentum up) with daily uptrend and volume confirmation
            if (slope_3_values[i] > 0 and 
                close[i] > ema50_1d_aligned[i] and  # daily uptrend
                volume[i] > vol_ma_20[i]):  # volume spike
                signals[i] = 0.25
                position = 1
            # Short conditions: negative slope (momentum down) with daily downtrend and volume confirmation
            elif (slope_3_values[i] < 0 and 
                  close[i] < ema50_1d_aligned[i] and  # daily downtrend
                  volume[i] > vol_ma_20[i]):  # volume spike
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long when slope turns negative or breaks against trend
            if (slope_3_values[i] < 0 or 
                close[i] < ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short when slope turns positive or breaks against trend
            if (slope_3_values[i] > 0 or 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals