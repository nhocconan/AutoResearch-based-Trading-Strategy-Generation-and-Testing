# -*- coding: utf-8 -*-
#!/usr/bin/env python3

# Hypothesis: Weekly 12-hour EMA trend filter combined with daily price action
# and volume confirmation on daily timeframe. Uses weekly trend to avoid
# counter-trend trades, and daily price/volume for precise entries.
# Designed for low trade frequency (<25/year) to minimize fee drag and
# work in both bull (trend following) and bear (avoiding false signals) markets.
# Target: Weekly EMA50 trend + Daily close > Weekly EMA + Volume spike.

name = "1d_WeeklyEMA50_Trend_Volume"
timeframe = "1d"
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
    
    # Get weekly data for trend filter
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 50:
        return np.zeros(n)
    
    # Weekly EMA50 for trend filter
    weekly_close = df_weekly['close'].values
    weekly_ema50 = pd.Series(weekly_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    weekly_ema50_aligned = align_htf_to_ltf(prices, df_weekly, weekly_ema50)
    
    # Daily volume filter: current volume > 2.0 * 20-day average
    vol_avg = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (vol_avg * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 50)  # Ensure enough data for weekly EMA50
    
    for i in range(start_idx, n):
        if np.isnan(weekly_ema50_aligned[i]) or np.isnan(vol_avg[i]):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: close above weekly EMA50 + volume spike
            if close[i] > weekly_ema50_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: close below weekly EMA50 + volume spike
            elif close[i] < weekly_ema50_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position != 0:
            # Exit: price crosses back through weekly EMA50
            if position == 1:
                if close[i] < weekly_ema50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            else:  # position == -1
                if close[i] > weekly_ema50_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals