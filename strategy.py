#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using 1-week EMA10 trend filter and 20-day Donchian channel breakout with volume confirmation.
# Long when price breaks above Donchian upper (20) with 1-week EMA10 uptrend and volume > 1.5x 20-day average.
# Short when price breaks below Donchian lower (20) with 1-week EMA10 downtrend and volume > 1.5x 20-day average.
# Exit when price crosses the Donchian middle (mean of upper and lower).
# Position size: 0.25 (25%).
# Target: ~20-40 trades/year to avoid fee drag. Uses weekly trend to filter noise in daily bars.

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1w data for EMA10 trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate Donchian Channel (20)
    dc_period = 20
    
    # Upper band = highest high of last 20 periods
    dc_upper = np.full(n, np.nan)
    for i in range(dc_period - 1, n):
        dc_upper[i] = np.max(high[i - dc_period + 1:i + 1])
    
    # Lower band = lowest low of last 20 periods
    dc_lower = np.full(n, np.nan)
    for i in range(dc_period - 1, n):
        dc_lower[i] = np.min(low[i - dc_period + 1:i + 1])
    
    # Middle band = average of upper and lower
    dc_middle = (dc_upper + dc_lower) / 2
    
    # Calculate 1-week EMA10 for trend filter
    ema_period = 10
    ema_1w = np.full(len(close_1w), np.nan)
    if len(close_1w) >= ema_period:
        ema_1w[ema_period - 1] = np.mean(close_1w[:ema_period])
        for i in range(ema_period, len(close_1w)):
            ema_1w[i] = (close_1w[i] * (2 / (ema_period + 1)) + 
                         ema_1w[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Get volume MA for confirmation
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align 1-week EMA10 to daily timeframe
    ema_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian(20), EMA10, and volume MA20
    start_idx = max(dc_period - 1, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(ema_1w_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: break above Donchian upper with 1w EMA10 uptrend and volume
            if (price > dc_upper[i] and 
                price > ema_1w_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: break below Donchian lower with 1w EMA10 downtrend and volume
            elif (price < dc_lower[i] and 
                  price < ema_1w_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian middle
            if price < dc_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian middle
            if price > dc_middle[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "1d_DonchianBreakout_1wEMA10_Volume"
timeframe = "1d"
leverage = 1.0