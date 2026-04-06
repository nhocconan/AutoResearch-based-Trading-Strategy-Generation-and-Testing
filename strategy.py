#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 1-day EMA trend filter and 1-week volume confirmation.
# Donchian breakouts capture trend continuation with clear entry/exit levels.
# EMA filter ensures trades align with higher timeframe trend.
# Volume confirmation filters weak breakouts.
# Designed for 4h timeframe targeting 75-200 trades over 4 years with moderate frequency.
# Works in bull markets via breakout continuation and bear markets via short breakdowns.

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
    
    # EMA calculation
    ema_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_1d[i] = close_1d[i] * 0.039216 + ema_1d[i-1] * 0.960784  # alpha = 2/(50+1)
    
    ema_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_1d)
    
    # 1-week volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    volume_1w = df_1w['volume'].values
    
    vol_ma_1w = np.full(len(volume_1w), np.nan)
    if len(volume_1w) >= 5:
        for i in range(4, len(volume_1w)):
            vol_ma_1w[i] = np.mean(volume_1w[i-4:i+1])
    
    vol_ma_1w_aligned = align_htf_to_ltf(prices, df_1w, vol_ma_1w)
    
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
    start = max(49, 4, 19)  # EMA needs 49, volume needs 4, Donchian needs 19
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_1d_aligned[i]) or np.isnan(vol_ma_1w_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x weekly average
        volume_filter = volume[i] > vol_ma_1w_aligned[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below Donchian low or stoploss
            if (close[i] < donchian_low[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above Donchian high or stoploss
            if (close[i] > donchian_high[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for breakouts with volume confirmation and trend filter
            if volume_filter:
                # Long: price breaks above Donchian high in uptrend
                if (close[i] > donchian_high[i] and close[i] > ema_1d_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below Donchian low in downtrend
                elif (close[i] < donchian_low[i] and close[i] < ema_1d_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals