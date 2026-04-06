#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian channel breakout (20-period) with 1-day EMA trend filter and volume confirmation.
# Donchian breakouts capture momentum in trending markets, EMA filter ensures alignment with higher timeframe trend,
# and volume confirmation reduces false breakouts. Designed for 4h timeframe targeting 75-200 trades over 4 years.
# Works in both bull and bear markets by filtering breakouts with trend and volume.

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
    
    # 1-day EMA(50) for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on daily close
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])  # Simple average for first value
        for i in range(50, len(close_1d)):
            ema_1d[i] = (close_1d[i] * 2 / (50 + 1)) + (ema_1d[i-1] * (49 / (50 + 1)))
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1-day volume average for confirmation
    vol_1d = df_1d['volume'].values
    vol_ma_1d = np.full(len(vol_1d), np.nan)
    if len(vol_1d) >= 20:
        for i in range(19, len(vol_1d)):
            vol_ma_1d[i] = np.mean(vol_1d[i-19:i+1])  # 20-period SMA
    
    vol_ma_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 50, 20)  # Donchian needs 20, EMA needs 50, volume needs 20
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1d_aligned[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Donchian channel (20-period)
        highest_high = np.max(high[i-20:i+1])
        lowest_low = np.min(low[i-20:i+1])
        
        # Volume condition: current volume > 1.5x daily average
        volume_filter = volume[i] > vol_ma_1d_aligned[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian lower band or stoploss
            if (close[i] < lowest_low or 
                close[i] < entry_price - 2.5 * (highest_high - lowest_low)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian upper band or stoploss
            if (close[i] > highest_high or 
                close[i] > entry_price + 2.5 * (highest_high - lowest_low)):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with volume and trend filter
            if volume_filter:
                # Long: price breaks above upper band and close above daily EMA
                if close[i] > highest_high and close[i] > ema_1d_aligned[i]:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below lower band and close below daily EMA
                elif close[i] < lowest_low and close[i] < ema_1d_aligned[i]:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals