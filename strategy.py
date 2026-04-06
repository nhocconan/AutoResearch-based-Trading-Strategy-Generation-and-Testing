#!/usr/bin/env python3
"""
1h session-filtered breakout with 4h Donchian(20) and 1d EMA(20) trend
Hypothesis: Trade breakouts only during high-liquidity hours (08-20 UTC) using 4h Donchian channels for breakout levels and 1d EMA for trend filter. 1h timeframe allows precise entry timing while 4h/1d filters reduce whipsaw. Target: 100-200 total trades over 4 years.
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "1h_session_breakout_4h_donchian_1d_ema"
timeframe = "1h"
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
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian channels (20-period) on 4h
    donchian_high_4h = np.full(len(high_4h), np.nan)
    donchian_low_4h = np.full(len(low_4h), np.nan)
    for i in range(20, len(high_4h)):
        donchian_high_4h[i] = np.max(high_4h[i-20:i])
        donchian_low_4h[i] = np.min(low_4h[i-20:i])
    
    # Align Donchian levels to 1h timeframe
    donchian_high_aligned = align_htf_to_ltf(prices, df_4h, donchian_high_4h)
    donchian_low_aligned = align_htf_to_ltf(prices, df_4h, donchian_low_4h)
    
    # Get 1d data for EMA trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(20) on 1d
    def ema(arr, period):
        if len(arr) < period:
            return np.full_like(arr, np.nan)
        alpha = 2.0 / (period + 1)
        ema_val = np.full_like(arr, np.nan)
        ema_val[period-1] = np.mean(arr[:period])
        for i in range(period, len(arr)):
            ema_val[i] = alpha * arr[i] + (1 - alpha) * ema_val[i-1]
        return ema_val
    
    ema_20 = ema(close_1d, 20)
    ema_20_aligned = align_htf_to_ltf(prices, df_1d, ema_20)
    
    # Pre-compute session hours (08-20 UTC)
    hours = pd.DatetimeIndex(prices["open_time"]).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(donchian_high_aligned[i]) or 
            np.isnan(donchian_low_aligned[i]) or np.isnan(ema_20_aligned[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        in_session = 8 <= hour <= 20
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price closes below 1h Donchian low (using 4h levels) or stoploss hit
            if (close[i] < donchian_low_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price closes above 1h Donchian high (using 4h levels) or stoploss hit
            if (close[i] > donchian_high_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries only during session
            if in_session:
                # Long: price breaks above 4h Donchian high and above 1d EMA20
                if (close[i] > donchian_high_aligned[i] and 
                    close[i] > ema_20_aligned[i]):
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below 4h Donchian low and below 1d EMA20
                elif (close[i] < donchian_low_aligned[i] and 
                      close[i] < ema_20_aligned[i]):
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
                else:
                    signals[i] = 0.0
            else:
                signals[i] = 0.0
    
    return signals