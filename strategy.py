#!/usr/bin/env python3
"""
12h Donchian Breakout with 1d Trend Filter and Volume Spike.
Long when price breaks above 20-period Donchian high + daily trend up + volume spike.
Short when price breaks below 20-period Donchian low + daily trend down + volume spike.
Exit when price crosses back to opposite Donchian band or trend changes.
Designed for low frequency (12-37 trades/year) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    # Daily EMA(34) for trend filter
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Donchian channels (20-period) on 12h data
    donch_high = np.full(n, np.nan)
    donch_low = np.full(n, np.nan)
    for i in range(19, n):
        donch_high[i] = np.max(high[i-19:i+1])
        donch_low[i] = np.min(low[i-19:i+1])
    
    # Volume filter: volume > 1.5x average (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian (20), EMA (34), volume MA (20)
    start_idx = max(20, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(donch_high[i]) or np.isnan(donch_low[i]) or 
            np.isnan(ema_34_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current indicators
        donch_high_now = donch_high[i]
        donch_low_now = donch_low[i]
        trend = ema_34_aligned[i]
        
        # Volume filter: volume > 1.5x average
        vol_filter = vol_now > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Bull: break above Donchian high + daily trend up + volume spike
            if price_now > donch_high_now and close[i-1] <= donch_high[i-1] and trend > price_now * 0.98 and vol_filter:
                signals[i] = size
                position = 1
            # Bear: break below Donchian low + daily trend down + volume spike
            elif price_now < donch_low_now and close[i-1] >= donch_low[i-1] and trend < price_now * 1.02 and vol_filter:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses back below Donchian low or trend turns down
            if price_now < donch_low_now or trend < price_now * 0.98:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses back above Donchian high or trend turns up
            if price_now > donch_high_now or trend > price_now * 1.02:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_DonchianBreakout_DailyTrend_Volume"
timeframe = "12h"
leverage = 1.0