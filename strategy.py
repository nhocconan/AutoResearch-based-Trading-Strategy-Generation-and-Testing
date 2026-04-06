#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour EMA(21) trend filter and volume confirmation
# Donchian breakouts capture momentum in trending markets. EMA filter ensures trades align with higher timeframe trend.
# Volume confirmation adds confirmation of institutional participation. Designed for 4h timeframe to target 75-200 trades over 4 years.

name = "4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4-hour Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # 12-hour EMA(21) for trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 21:
        ema_12h[20] = np.mean(close_12h[:21])
        for i in range(21, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 + ema_12h[i-1] * 19) / 21
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    
    # 12-hour volume average (20-period) for confirmation
    volume_12h = df_12h['volume'].values
    vol_ma_12h = np.full(len(volume_12h), np.nan)
    if len(volume_12h) >= 20:
        vol_ma_12h[19] = np.mean(volume_12h[:20])
        for i in range(20, len(volume_12h)):
            vol_ma_12h[i] = (volume_12h[i] + vol_ma_12h[i-1] * 19) / 20
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(19, 20, 20)  # Donchian needs 19, EMA needs 20, volume MA needs 20
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_12h_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 12-hour average
        volume_filter = volume[i] > vol_ma_12h_aligned[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below lower Donchian band or stoploss
            if (close[i] < lowest_low[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above upper Donchian band or stoploss
            if (close[i] > highest_high[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: price breaks above/below Donchian bands with trend and volume
            if volume_filter:
                # Long: price breaks above upper Donchian band and price above 12h EMA
                if (close[i] > highest_high[i] and close[i] > ema_12h_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below lower Donchian band and price below 12h EMA
                elif (close[i] < lowest_low[i] and close[i] < ema_12h_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals