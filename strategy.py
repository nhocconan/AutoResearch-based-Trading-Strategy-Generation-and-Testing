#!/usr/bin/env python3
# 1d_weekly_ema_trend_volume_v2
# Hypothesis: Use weekly EMA trend filter with daily price action and volume confirmation.
# In strong weekly trend (price above/below weekly EMA), look for daily pullbacks to enter in trend direction.
# Volume confirms momentum. Designed to work in both bull and bear markets by following higher timeframe trend.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

name = "1d_weekly_ema_trend_volume_v2"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price data
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly EMA trend filter (21-period)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    ema_weekly = pd.Series(df_weekly['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily EMA for entry timing (21-period)
    ema_daily = pd.Series(close).ewm(span=21, adjust=False, min_periods=21).mean()
    
    # Volume filter: volume > 1.5x 20-day average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean()
    
    # Start from sufficient lookback
    start_idx = max(21, 20) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_weekly_aligned[i]) or np.isnan(ema_daily[i]) or 
            np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below daily EMA or weekly trend fails
            if close[i] < ema_daily[i] or close[i] < ema_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above daily EMA or weekly trend fails
            if close[i] > ema_daily[i] or close[i] > ema_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_filter:
                # Enter long: price above both EMAs (pullback in uptrend)
                if close[i] > ema_daily[i] and close[i] > ema_weekly_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short: price below both EMAs (pullback in downtrend)
                elif close[i] < ema_daily[i] and close[i] < ema_weekly_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals