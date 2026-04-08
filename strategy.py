#!/usr/bin/env python3
# 4h_triple_barrier_breakout_1d_trend_volume_v1
# Hypothesis: Combine Donchian channel breakout with daily trend filter and volume confirmation.
# Enter long when price breaks above Donchian(20) high with daily uptrend and volume spike.
# Enter short when price breaks below Donchian(20) low with daily downtrend and volume spike.
# Exit when price touches opposite Donchian band or trend fails.
# Uses 4h timeframe with 1d trend filter to target 75-200 trades over 4 years (19-50/year).

name = "4h_triple_barrier_breakout_1d_trend_volume_v1"
timeframe = "4h"
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
    
    # Donchian Channel (20-period)
    dc_period = 20
    dc_high = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    dc_low = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    dc_mid = (dc_high + dc_low) / 2
    
    # Daily EMA trend filter (50-period)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 1:
        return np.zeros(n)
    
    ema_daily = pd.Series(df_daily['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Volume filter: volume > 1.5x 24-period average (4 days)
    vol_period = 24
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(dc_period, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or np.isnan(dc_mid[i]) or
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price touches middle band or trend fails
            if close[i] <= dc_mid[i] or close[i] <= ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price touches middle band or trend fails
            if close[i] >= dc_mid[i] or close[i] >= ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Require volume confirmation
            if volume_filter:
                # Breakout long: price breaks above Donchian high with uptrend
                if close[i] > dc_high[i] and close[i] > ema_daily_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price breaks below Donchian low with downtrend
                elif close[i] < dc_low[i] and close[i] < ema_daily_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals