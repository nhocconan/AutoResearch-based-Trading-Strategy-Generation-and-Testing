#!/usr/bin/env python3
# 4h_donchian_breakout_12h_trend_volume_v1
# Hypothesis: 4h Donchian channel breakout with 12h trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high with 12h uptrend and volume > 1.5x average.
# Short when price breaks below Donchian(20) low with 12h downtrend and volume > 1.5x average.
# Uses volume filter to reduce false signals and trend filter to avoid counter-trend trades.
# Target: 20-50 trades/year per symbol to minimize fee drag.

name = "4h_donchian_breakout_12h_trend_volume_v1"
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
    
    # Donchian channel (20-period)
    lookback = 20
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(lookback - 1, n):
        donchian_high[i] = np.max(high[i - lookback + 1:i + 1])
        donchian_low[i] = np.min(low[i - lookback + 1:i + 1])
    
    # Volume average (20-period)
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i - 19:i + 1])
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # 12h EMA (34-period) for trend
    ema_period = 34
    ema_12h = np.full(len(close_12h), np.nan)
    for i in range(ema_period - 1, len(close_12h)):
        if i == ema_period - 1:
            ema_12h[i] = np.mean(close_12h[:ema_period])
        else:
            ema_12h[i] = (close_12h[i] * 2 + ema_12h[i-1] * (ema_period - 1)) / (ema_period + 1)
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Pre-compute session filter (08-20 UTC)
    hours = pd.DatetimeIndex(prices['open_time']).hour
    in_session = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    # Start from sufficient lookback
    start_idx = max(lookback, 19, ema_period) + 5
    
    for i in range(start_idx, n):
        # Skip if outside trading session
        if not in_session[i]:
            signals[i] = 0.0
            continue
        
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter: current volume > 1.5x 20-period average
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        # 12h trend filter
        uptrend_12h = close[i] > ema_12h_aligned[i]
        downtrend_12h = close[i] < ema_12h_aligned[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian low or trend fails
            if close[i] < donchian_low[i] or not uptrend_12h:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian high or trend fails
            if close[i] > donchian_high[i] or not downtrend_12h:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: break above Donchian high, volume confirmation, 12h uptrend
            if (close[i] > donchian_high[i] and 
                volume_filter and 
                uptrend_12h):
                position = 1
                signals[i] = 0.25
            # Short entry: break below Donchian low, volume confirmation, 12h downtrend
            elif (close[i] < donchian_low[i] and 
                  volume_filter and 
                  downtrend_12h):
                position = -1
                signals[i] = -0.25
    
    return signals