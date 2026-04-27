#!/usr/bin/env python3
"""
Hypothesis: 4h Donchian breakout (20-period) with 1d trend filter and volume confirmation.
Enters long when price breaks above 4h Donchian upper band with above-average volume and daily uptrend.
Enters short when price breaks below 4h Donchian lower band with above-average volume and daily downtrend.
Uses daily timeframe for trend and volume context, 4h for execution.
Designed to work in both bull and bear markets by following the daily trend and requiring volume confirmation.
Target: 15-30 trades/year per symbol (60-120 total over 4 years) to minimize fee drag.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for trend and volume context
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate daily EMA(34) for trend filter
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate daily volume average for volume context
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Calculate 4h Donchian channels (20-period)
    donchian_period = 20
    upper = pd.Series(high).rolling(window=donchian_period, min_periods=donchian_period).max().values
    lower = pd.Series(low).rolling(window=donchian_period, min_periods=donchian_period).min().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian channels, daily EMA, and daily volume average
    start_idx = max(donchian_period, 34, 20)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper[i]) or np.isnan(lower[i]) or 
            np.isnan(ema_34_1d_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Current price and volume
        price_now = close[i]
        vol_now = volume[i]
        
        # Current Donchian levels
        upper_now = upper[i]
        lower_now = lower[i]
        
        # Daily trend and volume context
        trend_1d = ema_34_1d_aligned[i]
        vol_avg = vol_avg_1d_aligned[i]
        
        # Volume filter: volume > 1.5x daily average
        vol_filter = vol_now > 1.5 * vol_avg
        
        # Entry conditions: Donchian breakout with volume and daily trend alignment
        if position == 0:
            # Long: price breaks above upper band with volume + daily uptrend
            if price_now > upper_now and vol_filter and price_now > trend_1d:
                signals[i] = size
                position = 1
            # Short: price breaks below lower band with volume + daily downtrend
            elif price_now < lower_now and vol_filter and price_now < trend_1d:
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price returns to lower band or daily trend turns down
            if price_now <= lower_now or price_now < trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price returns to upper band or daily trend turns up
            if price_now >= upper_now or price_now > trend_1d:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0