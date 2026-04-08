#!/usr/bin/env python3
# 1d_weekly_ema_trend_volume_v1
# Hypothesis: Combine weekly EMA trend filter with daily price action and volume confirmation.
# Enter long when daily price crosses above weekly EMA with volume confirmation.
# Enter short when daily price crosses below weekly EMA with volume confirmation.
# Uses weekly trend to avoid whipsaw in ranging markets and capture directional moves.
# Target: 30-100 total trades over 4 years (7-25/year) to minimize fee drag.

name = "1d_weekly_ema_trend_volume_v1"
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
    volume = prices['volume'].values
    
    # Weekly EMA trend filter (21-period)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 1:
        return np.zeros(n)
    
    ema_weekly = pd.Series(df_weekly['close'].values).ewm(span=21, adjust=False, min_periods=21).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Daily EMA for dynamic support/resistance (50-period)
    ema_daily = pd.Series(close).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Volume filter: volume > 1.5x 20-day average
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(21, 50, vol_period) + 5
    
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
            # Exit: price crosses below weekly EMA or daily EMA
            if close[i] < ema_weekly_aligned[i] or close[i] < ema_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price crosses above weekly EMA or daily EMA
            if close[i] > ema_weekly_aligned[i] or close[i] > ema_daily[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_filter:
                # Entry long: price crosses above weekly EMA with bullish alignment
                if close[i] > ema_weekly_aligned[i] and close[i] > ema_daily[i]:
                    position = 1
                    signals[i] = 0.25
                # Entry short: price crosses below weekly EMA with bearish alignment
                elif close[i] < ema_weekly_aligned[i] and close[i] < ema_daily[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals