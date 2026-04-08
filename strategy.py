#!/usr/bin/env python3
# 12h_donchian_breakout_1d_trend_volume_v1
# Hypothesis: 12h Donchian breakout (20-period) with volume confirmation and daily trend filter.
# Enters long on breakout above upper band in daily uptrend, short on breakdown below lower band in daily downtrend.
# Volume filter (>1.5x average) reduces false signals. Targets 20-40 trades/year to minimize fee drag.

name = "12h_donchian_breakout_1d_trend_volume_v1"
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
    
    # Donchian channels (20-period)
    period = 20
    upper = np.full_like(high, np.nan)
    lower = np.full_like(low, np.nan)
    
    for i in range(period-1, len(high)):
        upper[i] = np.max(high[i-period+1:i+1])
        lower[i] = np.min(low[i-period+1:i+1])
    
    # Volume filter: 20-period average volume
    vol_ma = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    # Get daily data for trend filter
    df_daily = get_htf_data(prices, '1d')
    close_daily = df_daily['close'].values
    
    # Daily EMA (50-period) for higher timeframe trend
    ema_period = 50
    ema_daily = np.full_like(close_daily, np.nan)
    for i in range(ema_period-1, len(close_daily)):
        ema_daily[i] = np.mean(close_daily[i-ema_period+1:i+1])
    
    # Align daily EMA to 12h timeframe
    ema_daily_aligned = align_htf_to_ltf(prices, df_daily, ema_daily)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(period, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_daily_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # Higher timeframe trend filter: price above/below daily EMA
        uptrend_htf = close[i] > ema_daily_aligned[i]
        downtrend_htf = close[i] < ema_daily_aligned[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below lower band or volume fails
            if close[i] < lower[i] or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above upper band or volume fails
            if close[i] > upper[i] or not volume_filter:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: breakout above upper band, volume confirmation, and daily uptrend
            if (close[i] > upper[i] and 
                volume_filter and 
                uptrend_htf):
                position = 1
                signals[i] = 0.25
            # Short entry: breakdown below lower band, volume confirmation, and daily downtrend
            elif (close[i] < lower[i] and 
                  volume_filter and 
                  downtrend_htf):
                position = -1
                signals[i] = -0.25
    
    return signals