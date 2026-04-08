#!/usr/bin/env python3
# 12h_Donchian_1d_trend_volume_v1
# Hypothesis: 12h Donchian breakout with 1d EMA trend filter and volume confirmation. Works in bull/bear by following higher timeframe trend. Target: 15-30 trades/year to minimize fee drift.

name = "12h_Donchian_1d_trend_volume_v1"
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
    
    # 12h Donchian channels (20-period)
    donchian_period = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    for i in range(donchian_period-1, n):
        donchian_high[i] = np.max(high[i-donchian_period+1:i+1])
        donchian_low[i] = np.min(low[i-donchian_period+1:i+1])
    
    # 1d EMA trend filter (34-period)
    df_daily = get_htf_data(prices, '1d')
    if len(df_daily) < 34:
        return np.zeros(n)
    ema_daily = pd.Series(df_daily['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Volume filter: volume > 2.0x 30-period average (~5 days)
    vol_period = 30
    vol_ma = np.full(n, np.nan)
    vol_ma[vol_period-1:] = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()[vol_period-1:].values
    
    # Start from sufficient lookback
    start_idx = max(donchian_period, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 2.0 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit: price closes below 12h Donchian low or trend fails
            if close[i] < donchian_low[i] or close[i] < ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit: price closes above 12h Donchian high or trend fails
            if close[i] > donchian_high[i] or close[i] > ema_daily_aligned[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Only trade with volume confirmation
            if volume_filter:
                # Breakout long: price breaks above 12h Donchian high with uptrend
                if close[i] > donchian_high[i] and close[i] > ema_daily_aligned[i]:
                    position = 1
                    signals[i] = 0.25
                # Breakout short: price breaks below 12h Donchian low with downtrend
                elif close[i] < donchian_low[i] and close[i] < ema_daily_aligned[i]:
                    position = -1
                    signals[i] = -0.25
    
    return signals