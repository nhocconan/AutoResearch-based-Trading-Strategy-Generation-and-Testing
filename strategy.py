#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with 1d EMA50 trend filter and volume confirmation, with ATR-based stoploss.
# Long when price breaks above Donchian(20) high with 1d EMA50 uptrend and volume > 1.5x average.
# Short when price breaks below Donchian(20) low with 1d EMA50 downtrend and volume > 1.5x average.
# Exit when price closes below Donchian(10) high (long) or above Donchian(10) low (short).
# Focus on high-probability breakouts in trending markets to minimize trades and maximize edge.

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
    donchian_period_entry = 20
    donchian_period_exit = 10
    upper_entry = np.full(n, np.nan)
    lower_entry = np.full(n, np.nan)
    upper_exit = np.full(n, np.nan)
    lower_exit = np.full(n, np.nan)
    
    for i in range(donchian_period_entry - 1, n):
        upper_entry[i] = np.max(high[i - donchian_period_entry + 1:i + 1])
        lower_entry[i] = np.min(low[i - donchian_period_entry + 1:i + 1])
    
    for i in range(donchian_period_exit - 1, n):
        upper_exit[i] = np.max(high[i - donchian_period_exit + 1:i + 1])
        lower_exit[i] = np.min(low[i - donchian_period_exit + 1:i + 1])
    
    # Calculate ATR (20-period) for stoploss and volatility filter
    atr_period = 20
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    
    atr = np.full(n, np.nan)
    for i in range(atr_period - 1, n):
        atr[i] = np.mean(tr[i - atr_period + 1:i + 1])
    
    # Volume MA for confirmation (20-period)
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    # Align 1d EMA50 to 4h timeframe
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian(20), EMA50, ATR(20), and volume MA20
    start_idx = max(donchian_period_entry, ema_period - 1, atr_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_entry[i]) or np.isnan(lower_entry[i]) or 
            np.isnan(ema_1d_aligned[i]) or np.isnan(atr[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Volume filter: require volume above average
        vol_filter = vol_now > 1.5 * vol_avg
        
        if position == 0:
            # Long: break above Donchian(20) high with 1d EMA50 uptrend and volume filter
            if (price > upper_entry[i] and 
                price > ema_1d_aligned[i] and vol_filter):
                signals[i] = size
                position = 1
            # Short: break below Donchian(20) low with 1d EMA50 downtrend and volume filter
            elif (price < lower_entry[i] and 
                  price < ema_1d_aligned[i] and vol_filter):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: close below Donchian(10) high or stoploss
            if price < upper_exit[i] or price < close[i-1] - 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: close above Donchian(10) low or stoploss
            if price > lower_exit[i] or price > close[i-1] + 2.5 * atr[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "4h_Donchian20_Breakout_1dEMA50_Volume_ATR"
timeframe = "4h"
leverage = 1.0