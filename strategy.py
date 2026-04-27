#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Donchian channel breakout with 1d EMA50 trend filter and volume confirmation.
# Long when price breaks above 20-period Donchian upper band with 1d EMA50 uptrend and volume > 1.5x average.
# Short when price breaks below 20-period Donchian lower band with 1d EMA50 downtrend and volume > 1.5x average.
# Exit when price crosses the 10-period Donchian midpoint (mean reversion).
# Uses Donchian channels for clear breakout signals, targeting 12-37 trades per year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate Donchian channels (20-period for entry, 10-period for exit)
    donch_period_entry = 20
    donch_period_exit = 10
    
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    upper_band_exit = np.full(n, np.nan)
    lower_band_exit = np.full(n, np.nan)
    
    for i in range(donch_period_entry - 1, n):
        upper_band[i] = np.max(high[i - donch_period_entry + 1:i + 1])
        lower_band[i] = np.min(low[i - donch_period_entry + 1:i + 1])
    
    for i in range(donch_period_exit - 1, n):
        upper_band_exit[i] = np.max(high[i - donch_period_exit + 1:i + 1])
        lower_band_exit[i] = np.min(low[i - donch_period_exit + 1:i + 1])
    
    # Exit midpoint (average of upper and lower bands for 10-period)
    midpoint_exit = (upper_band_exit + lower_band_exit) / 2.0
    
    # Align 1d EMA to 12h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian channels, EMA50, and volume MA20
    start_idx = max(donch_period_entry, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(midpoint_exit[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above 20-period upper band with 1d EMA50 uptrend and volume filter
            if (price > upper_band[i] and 
                price > ema_1d_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below 20-period lower band with 1d EMA50 downtrend and volume filter
            elif (price < lower_band[i] and 
                  price < ema_1d_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below 10-period midpoint
            if price < midpoint_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above 10-period midpoint
            if price > midpoint_exit[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "12h_Donchian20_Breakout_1dEMA50_Volume"
timeframe = "12h"
leverage = 1.0