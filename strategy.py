#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout (20-period) with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel with 12h EMA50 uptrend and volume > 1.8x average.
# Short when price breaks below lower Donchian channel with 12h EMA50 downtrend and volume > 1.8x average.
# Exit when price crosses back through Donchian middle (mean reversion) or ATR-based stoploss.
# Focus on high-probability breakouts with trend alignment to minimize false signals and control trade frequency.

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
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    middle_channel = np.full(n, np.nan)
    
    for i in range(donch_period - 1, n):
        upper_channel[i] = np.max(high[i - donch_period + 1:i + 1])
        lower_channel[i] = np.min(low[i - donch_period + 1:i + 1])
        middle_channel[i] = (upper_channel[i] + lower_channel[i]) / 2
    
    # Align 12h EMA to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # ATR for stoploss (14-period)
    atr_period = 14
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    for i in range(atr_period - 1, n):
        if i == atr_period - 1:
            atr[i] = np.mean(tr[1:i+1])
        else:
            atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian, EMA12h, volume MA20, ATR
    start_idx = max(donch_period, ema_period - 1, 19, atr_period - 1)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(middle_channel[i]) or np.isnan(ema_12h_aligned[i]) or 
            np.isnan(vol_ma_20[i]) or np.isnan(atr[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume significantly above average
        vol_filter = vol_now > 1.8 * vol_avg
        
        if position == 0:
            # Long: price breaks above upper Donchian with 12h EMA50 uptrend and volume filter
            if (price > upper_channel[i] and 
                price > ema_12h_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: price breaks below lower Donchian with 12h EMA50 downtrend and volume filter
            elif (price < lower_channel[i] and 
                  price < ema_12h_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit conditions: price crosses below middle OR stoploss hit
            exit_signal = (price < middle_channel[i] or 
                          price <= close[i-1] - 2.0 * atr[i-1])
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit conditions: price crosses above middle OR stoploss hit
            exit_signal = (price > middle_channel[i] or 
                          price >= close[i-1] + 2.0 * atr[i-1])
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_12hEMA50_Volume"
timeframe = "4h"
leverage = 1.0