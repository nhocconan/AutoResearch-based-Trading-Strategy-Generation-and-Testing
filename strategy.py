#!/usr/bin/env python3
"""
1h Camarilla H3L3 Breakout + 4h EMA50 Trend + Volume Spike
Hypothesis: 1h Camarilla H3/L3 breakouts with volume confirmation and 4h EMA50 trend alignment capture intraday momentum.
Designed for 60-150 trades over 4 years (15-37/year) on 1h timeframe. Uses 4h for signal direction, 1h for entry timing.
Includes session filter (08-20 UTC) to reduce noise. Fixed size 0.20 to control drawdown.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def calculate_ema(series, period):
    """Calculate Exponential Moving Average with min_periods"""
    if len(series) < period:
        return np.full_like(series, np.nan)
    return pd.Series(series).ewm(span=period, adjust=False, min_periods=period).mean().values

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h data for EMA50 trend filter (loaded ONCE)
    df_4h = get_htf_data(prices, '4h')
    ema_50_4h = calculate_ema(df_4h['close'].values, 50)
    ema_50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema_50_4h)
    
    # 1d data for daily Camarilla pivots (using previous day)
    df_1d = get_htf_data(prices, '1d')
    prev_daily_high = np.roll(df_1d['high'].values, 1)
    prev_daily_low = np.roll(df_1d['low'].values, 1)
    prev_daily_close = np.roll(df_1d['close'].values, 1)
    prev_daily_high[0] = np.nan
    prev_daily_low[0] = np.nan
    prev_daily_close[0] = np.nan
    
    prev_daily_range = prev_daily_high - prev_daily_low
    # Daily Camarilla levels
    daily_h3 = prev_daily_close + 1.1 * prev_daily_range / 4  # H3
    daily_l3 = prev_daily_close - 1.1 * prev_daily_range / 4  # L3
    daily_h3_aligned = align_htf_to_ltf(prices, df_1d, daily_h3)
    daily_l3_aligned = align_htf_to_ltf(prices, df_1d, daily_l3)
    
    # Volume confirmation: current volume > 2.0 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_spike = volume > (vol_ma * 2.0)
    
    # Session filter: 08-20 UTC (pre-compute hour array)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start index: need enough for EMA, volume MA
    start_idx = max(50, 20) + 5
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(ema_50_4h_aligned[i]) or np.isnan(daily_h3_aligned[i]) or np.isnan(daily_l3_aligned[i]) or
            np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        # Session filter
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        curr_close = close[i]
        curr_high = high[i]
        curr_low = low[i]
        vol_spike = volume_spike[i]
        
        # Breakout conditions using daily levels
        breakout_long = curr_close > daily_h3_aligned[i]  # Break above daily H3
        breakout_short = curr_close < daily_l3_aligned[i]  # Break below daily L3
        
        if position == 0:
            # Look for entry signals - require: Daily Camarilla breakout + volume spike + 4h EMA50 trend alignment
            long_entry = breakout_long and vol_spike and (curr_close > ema_50_4h_aligned[i])
            short_entry = breakout_short and vol_spike and (curr_close < ema_50_4h_aligned[i])
            
            if long_entry:
                signals[i] = 0.20
                position = 1
            elif short_entry:
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Long position: exit on trend change or retrace to L3
            if curr_close < ema_50_4h_aligned[i] or curr_close < daily_l3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short position: exit on trend change or retrace to H3
            if curr_close > ema_50_4h_aligned[i] or curr_close > daily_h3_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals

name = "1h_Camarilla_H3L3_Breakout_4hEMA50_Trend_VolumeSpike_SessionFilter"
timeframe = "1h"
leverage = 1.0