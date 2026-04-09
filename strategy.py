#!/usr/bin/env python3
# 1h_4d_ema_trend_v1
# Hypothesis: 1-hour trend following using 4-hour EMA(20) direction with 1-hour EMA(20) pullback entries.
# Uses 4h EMA(20) for trend direction and 1h EMA(20) for entry timing on pullbacks.
# Includes volume confirmation (volume > 1.5x 20-period average) and session filter (08-20 UTC).
# Works in both bull and bear markets by following the 4h trend and entering on pullbacks.
# Target: 60-150 total trades over 4 years (15-37/year) to avoid fee drag.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_4d_ema_trend_v1"
timeframe = "1h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate 1h EMA(20) for entry timing
    close_s = pd.Series(close)
    ema_20 = close_s.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Load 4h data ONCE before loop for trend direction
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 2:
        return np.zeros(n)
    
    # Calculate 4h EMA(20) for trend direction
    close_4h = df_4h['close'].values
    ema_20_4h = pd.Series(close_4h).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_20_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_20_4h)
    
    # Volume confirmation: volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    vol_sum = 0
    for i in range(n):
        vol_sum += volume[i]
        if i >= 20:
            vol_sum -= volume[i-20]
        if i >= 19:
            vol_ma_20[i] = vol_sum / 20
    
    # Session filter: 08-20 UTC
    hours = prices.index.hour
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(50, n):  # Start after warmup
        # Skip if any required data is invalid
        if np.isnan(ema_20[i]) or np.isnan(ema_20_4h_aligned[i]) or np.isnan(vol_ma_20[i]):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade between 08:00-20:00 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume > 1.5x 20-period average
        vol_ok = volume[i] > vol_ma_20[i] * 1.5
        
        if position == 1:  # Long position
            # Exit: price crosses below 1h EMA(20) or 4h trend turns bearish
            if close[i] < ema_20[i] or ema_20_4h_aligned[i] > ema_20_4h_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.20
                
        elif position == -1:  # Short position
            # Exit: price crosses above 1h EMA(20) or 4h trend turns bullish
            if close[i] > ema_20[i] or ema_20_4h_aligned[i] < ema_20_4h_aligned[i-1]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.20
        else:  # Flat
            # Enter long: price above 1h EMA(20) AND 4h EMA trending up AND volume confirmation
            if close[i] > ema_20[i] and ema_20_4h_aligned[i] > ema_20_4h_aligned[i-1] and vol_ok:
                position = 1
                signals[i] = 0.20
            # Enter short: price below 1h EMA(20) AND 4h EMA trending down AND volume confirmation
            elif close[i] < ema_20[i] and ema_20_4h_aligned[i] < ema_20_4h_aligned[i-1] and vol_ok:
                position = -1
                signals[i] = -0.20
    
    return signals