#!/usr/bin/env python3
# 1d_weekly_ema_trend_volume_v5
# Hypothesis: Weekly EMA trend filter with daily price action and volume confirmation.
# Uses weekly EMA(34) to determine trend direction, enters on daily close in trend direction
# with volume > 1.5x average. Exits when price closes opposite weekly EMA.
# Designed for low trade frequency (<15/year) to minimize fee drift in 1d timeframe.

name = "1d_weekly_ema_trend_volume_v5"
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
    
    # Weekly EMA trend filter (34-period)
    df_weekly = get_htf_data(prices, '1w')
    if len(df_weekly) < 34:
        return np.zeros(n)
    ema_weekly = pd.Series(df_weekly['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_weekly_aligned = align_htf_to_ltf(prices, df_weekly, ema_weekly)
    
    # Volume filter: volume > 1.5x 20-period average (~1 month)
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(34, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema_weekly_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below weekly EMA
            if close[i] < ema_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above weekly EMA
            if close[i] > ema_weekly_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_filter:
                # Enter long: price closes above weekly EMA with volume
                if close[i] > ema_weekly_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Enter short: price closes below weekly EMA with volume
                elif close[i] < ema_weekly_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals