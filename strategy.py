#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian(20) breakout with 1d EMA(20) trend filter and volume confirmation.
# Long when price breaks above Donchian high with price above 1d EMA and volume > 1.5x average.
# Short when price breaks below Donchian low with price below 1d EMA and volume > 1.5x average.
# Uses tight entry conditions to target 75-200 trades over 4 years with low frequency.

name = "4h_donchian20_1d_ema_vol_v1"
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
    
    # 1-day EMA(20) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # EMA calculation
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 20:
        ema_1d[19] = np.mean(close_1d[:20])
        for i in range(20, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 + ema_1d[i-1] * 18) / 20
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1-day volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    for i in range(19, len(vol_1d)):  # 20-period average
        vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])
    
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    # Donchian channels (20-period)
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = 20
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below Donchian low or stoploss
            if (close[i] < donchian_low[i] or 
                close[i] < entry_price - 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above Donchian high or stoploss
            if (close[i] > donchian_high[i] or 
                close[i] > entry_price + 2.5 * (high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            if volume_filter:
                # Long: price breaks above Donchian high with price above EMA
                if (close[i] > donchian_high[i] and close[i] > ema_1d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below Donchian low with price below EMA
                elif (close[i] < donchian_low[i] and close[i] < ema_1d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals