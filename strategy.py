#!/usr/bin/env python3
# 4h_donchian_breakout_12h_trend_volume_v1
# Hypothesis: 4h Donchian channel breakout with 12h trend filter and volume confirmation.
# Uses Donchian(20) breakout in direction of 12h EMA(20) trend, filtered by volume > 1.5x average.
# Designed to capture strong trends while minimizing false breakouts in choppy markets.
# Target: 20-40 trades/year with ~0.25 position size to minimize fee drag.

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
    
    # Donchian Channel (20)
    dc_period = 20
    dc_upper = np.full(n, np.nan)
    dc_lower = np.full(n, np.nan)
    
    for i in range(dc_period - 1, n):
        dc_upper[i] = np.max(high[i - dc_period + 1:i + 1])
        dc_lower[i] = np.min(low[i - dc_period + 1:i + 1])
    
    # 12h EMA trend filter
    df_12h = get_htf_data(prices, '12h')
    ema_period = 20
    ema_12h = np.full(len(df_12h), np.nan)
    if len(df_12h) >= ema_period:
        ema_series = pd.Series(df_12h['close'].values).ewm(span=ema_period, adjust=False).mean()
        ema_12h = ema_series.values
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_period = 20
    vol_ma = np.full(n, np.nan)
    if n >= vol_period:
        vol_series = pd.Series(volume).rolling(window=vol_period, min_periods=vol_period).mean()
        vol_ma = vol_series.values
    
    # Start from sufficient lookback
    start_idx = max(dc_period, vol_period) + 5
    
    signals = np.zeros(n)
    position = 0  # 1=long, -1=short, 0=flat
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma[i]) or volume[i] == 0):
            signals[i] = 0.0
            continue
        
        # Volume filter
        volume_filter = volume[i] > 1.5 * vol_ma[i]
        
        if position == 1:  # Long position
            # Exit if price breaks below Donchian lower or trend turns bearish
            if close[i] < dc_lower[i] or ema_12h_aligned[i] > close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = 0.25
                
        elif position == -1:  # Short position
            # Exit if price breaks above Donchian upper or trend turns bullish
            if close[i] > dc_upper[i] or ema_12h_aligned[i] < close[i]:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -0.25
        else:  # Flat, look for entry
            # Long entry: break above upper Donchian with bullish trend and volume
            if close[i] > dc_upper[i] and ema_12h_aligned[i] < close[i] and volume_filter:
                position = 1
                signals[i] = 0.25
            # Short entry: break below lower Donchian with bearish trend and volume
            elif close[i] < dc_lower[i] and ema_12h_aligned[i] > close[i] and volume_filter:
                position = -1
                signals[i] = -0.25
    
    return signals