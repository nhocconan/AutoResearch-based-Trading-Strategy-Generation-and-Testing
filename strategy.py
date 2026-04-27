#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian upper band (20) with 12h EMA50 uptrend and volume > 1.5x average.
# Short when price breaks below Donchian lower band (20) with 12h EMA50 downtrend and volume > 1.5x average.
# Exit when price crosses back through Donchian middle band (20-period average).
# Uses Donchian channels for breakout detection, targeting 20-50 trades per year.

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 12h data for trend filter
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    
    # Calculate 12h EMA50 for trend filter
    ema_period = 50
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= ema_period:
        ema_12h[ema_period - 1] = np.mean(close_12h[:ema_period])
        for i in range(ema_period, len(close_12h)):
            ema_12h[i] = (close_12h[i] * (2 / (ema_period + 1)) + 
                         ema_12h[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate Donchian channels (20-period)
    donch_period = 20
    upper_band = np.full(n, np.nan)
    lower_band = np.full(n, np.nan)
    middle_band = np.full(n, np.nan)
    
    for i in range(donch_period - 1, n):
        upper_band[i] = np.max(high[i - donch_period + 1:i + 1])
        lower_band[i] = np.min(low[i - donch_period + 1:i + 1])
        middle_band[i] = (upper_band[i] + lower_band[i]) / 2
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian bands, EMA50, and volume MA20
    start_idx = max(donch_period - 1, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or np.isnan(middle_band[i]) or
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above upper band with 12h EMA50 uptrend and volume filter
            if (price > upper_band[i] and 
                price > ema_12h_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below lower band with 12h EMA50 downtrend and volume filter
            elif (price < lower_band[i] and 
                  price < ema_12h_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below middle band
            if price < middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above middle band
            if price > middle_band[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0