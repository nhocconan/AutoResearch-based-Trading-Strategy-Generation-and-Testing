#!/usr/bin/env python3
# 12h_donchian_20_1d_trend_volume_v1
# Hypothesis: Donchian(20) breakout with 1d EMA(50) trend filter and volume confirmation.
# Long when price breaks above 20-period high in uptrend with volume; short when breaks below 20-period low in downtrend with volume.
# Works in bull/bear: 1d EMA ensures directional alignment, Donchian captures breakouts, volume avoids false signals.
# Uses 12h timeframe for lower frequency, targeting 50-150 total trades over 4 years.

name = "12h_donchian_20_1d_trend_volume_v1"
timeframe = "12h"
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
    
    # 1d EMA trend filter (50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    ema_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Donchian channel (20-period) on 12h data
    period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    donchian_high[period-1:] = pd.Series(high).rolling(window=period, min_periods=period).max()[period-1:].values
    donchian_low[period-1:] = pd.Series(low).rolling(window=period, min_periods=period).min()[period-1:].values
    
    # Volume filter: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(period, 50) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below 20-period low or trend fails
            if close[i] < donchian_low[i] or close[i] < ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above 20-period high or trend fails
            if close[i] > donchian_high[i] or close[i] > ema_1d_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: price breaks above 20-period high with uptrend and volume
            if close[i] > donchian_high[i] and close[i] > ema_1d_aligned[i] and volume_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: price breaks below 20-period low with downtrend and volume
            elif close[i] < donchian_low[i] and close[i] < ema_1d_aligned[i] and volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals