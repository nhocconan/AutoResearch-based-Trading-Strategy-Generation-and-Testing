#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for trend filter and volatility
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate 1d ATR(14) for volatility filter
    atr_period = 14
    tr = np.zeros(len(close_1d))
    tr[0] = high_1d[0] - low_1d[0]
    for i in range(1, len(close_1d)):
        tr[i] = max(high_1d[i] - low_1d[i], 
                    abs(high_1d[i] - close_1d[i-1]), 
                    abs(low_1d[i] - close_1d[i-1]))
    
    atr_1d = np.full(len(close_1d), np.nan)
    if len(tr) >= atr_period:
        atr_1d[atr_period - 1] = np.mean(tr[:atr_period])
        for i in range(atr_period, len(tr)):
            atr_1d[i] = (tr[i] * (1 / atr_period) + 
                         atr_1d[i-1] * (1 - 1 / atr_period))
    
    # Calculate 1d EMA(50) for trend filter
    ema_period = 50
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= ema_period:
        ema_1d[ema_period - 1] = np.mean(close_1d[:ema_period])
        for i in range(ema_period, len(close_1d)):
            ema_1d[i] = (close_1d[i] * (2 / (ema_period + 1)) + 
                         ema_1d[i-1] * (1 - 2 / (ema_period + 1)))
    
    # Calculate 6-hour Donchian channels (20-period)
    donch_period = 20
    upper_channel = np.full(n, np.nan)
    lower_channel = np.full(n, np.nan)
    
    for i in range(donch_period - 1, n):
        upper_channel[i] = np.max(high[i - donch_period + 1:i + 1])
        lower_channel[i] = np.min(low[i - donch_period + 1:i + 1])
    
    # Align 1d indicators to 6h timeframe
    atr_1d_aligned = align_htf_to_ltf(prices, df_1d, atr_1d)
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # Volume filter: 6-hour volume > 1.5x 20-period average
    vol_ma_20 = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_20[i] = np.mean(volume[i - 19:i + 1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    size = 0.25   # 25% position size
    
    # Warmup: need Donchian channels, 1d indicators, and volume MA
    start_idx = max(donch_period - 1, atr_period - 1, ema_period - 1, 19)
    
    for i in range(start_idx, n):
        # Skip if any data not ready
        if (np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or 
            np.isnan(atr_1d_aligned[i]) or np.isnan(ema_1d_aligned[i]) or 
            np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        price = close[i]
        vol_now = volume[i]
        vol_avg = vol_ma_20[i]
        
        # Filters
        vol_filter = vol_now > 1.5 * vol_avg
        vol_filter_high = vol_now > 2.0 * vol_avg  # Strong volume for breakouts
        low_volatility = atr_1d_aligned[i] < np.nanpercentile(atr_1d_aligned[:i+1], 70) if i >= 20 else True
        
        if position == 0:
            # Long: Break above upper Donchian with strong volume, in uptrend, and not too volatile
            if (price > upper_channel[i] and 
                vol_filter_high and 
                price > ema_1d_aligned[i] and 
                low_volatility):
                signals[i] = size
                position = 1
            # Short: Break below lower Donchian with strong volume, in downtrend, and not too volatile
            elif (price < lower_channel[i] and 
                  vol_filter_high and 
                  price < ema_1d_aligned[i] and 
                  low_volatility):
                signals[i] = -size
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: Price crosses below 10-period EMA or stop loss
            ema_10 = np.full(n, np.nan)
            if i >= 9:
                ema_10[i] = (close[i] * (2 / 11) + 
                            (ema_10[i-1] if i > 0 and not np.isnan(ema_10[i-1]) else close[i]) * (9 / 11))
            
            if (i > 0 and not np.isnan(ema_10[i]) and 
                price < ema_10[i]) or \
               (price < close[i-1] - 2.0 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = size
        elif position == -1:
            # Exit short: Price crosses above 10-period EMA or stop loss
            ema_10 = np.full(n, np.nan)
            if i >= 9:
                ema_10[i] = (close[i] * (2 / 11) + 
                            (ema_10[i-1] if i > 0 and not np.isnan(ema_10[i-1]) else close[i]) * (9 / 11))
            
            if (i > 0 and not np.isnan(ema_10[i]) and 
                price > ema_10[i]) or \
               (price > close[i-1] + 2.0 * atr_1d_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -size
    
    return signals

name = "6h_Donchian20_Breakout_1dEMA50_ATR_Volume"
timeframe = "6h"
leverage = 1.0