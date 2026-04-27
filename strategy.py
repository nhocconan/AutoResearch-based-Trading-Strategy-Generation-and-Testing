#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above Donchian high (20) and 1d EMA34 is rising.
# Short when price breaks below Donchian low (20) and 1d EMA34 is falling.
# Exit when price crosses Donchian midline (10-period average of high/low).
# Uses price channel breakouts for trend following, targeting 20-50 trades per year.

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
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA34 for trend filter
    ema_period = 34
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i - 1] * (1 - (2 / (ema_period + 1))))
    
    # Calculate Donchian channels (20-period)
    donchian_period = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    middle_channel = np.full(n, np.nan)
    
    for i in range(donchian_period - 1, n):
        upper_channel[i] = np.max(high[i - donchian_period + 1:i + 1])
        lower_channel[i] = np.min(low[i - donchian_period + 1:i + 1])
        middle_channel[i] = (upper_channel[i] + lower_channel[i]) / 2
    
    # Calculate EMA of Donchian midline for trend filter
    ema_mid_period = 10
    ema_middle = np.full(n, np.nan)
    for i in range(ema_mid_period - 1, n):
        if i == ema_mid_period - 1:
            ema_middle[i] = np.nanmean(middle_channel[i - ema_mid_period + 1:i + 1])
        else:
            ema_middle[i] = (middle_channel[i] * (2 / (ema_mid_period + 1)) + 
                             ema_middle[i - 1] * (1 - (2 / (ema_mid_period + 1))))
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align 1d EMA to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian channels, EMA34, and volume MA20
    start_idx = max(donchian_period - 1, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_20[i]) or
            np.isnan(ema_middle[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: price breaks above upper Donchian channel with 1d EMA34 uptrend and volume filter
            if (price > upper_channel[i] and 
                ema_1d_aligned[i] > ema_1d_aligned[i-1] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian channel with 1d EMA34 downtrend and volume filter
            elif (price < lower_channel[i] and 
                  ema_1d_aligned[i] < ema_1d_aligned[i-1] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price crosses below Donchian midline
            if price < middle_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: price crosses above Donchian midline
            if price > middle_channel[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA34_Volume"
timeframe = "4h"
leverage = 1.0