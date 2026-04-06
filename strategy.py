#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian breakout with 1-day EMA trend filter and 1-week volume confirmation.
# Donchian(20) breakouts capture trend momentum with clear entry/exit rules.
# 1-day EMA(50) filters for trend direction to avoid counter-trend trades.
# 1-week volume > 1.5x average confirms institutional participation.
# Designed for 4h timeframe to target 75-200 trades over 4 years with moderate frequency.

name = "4h_donchian20_1d_ema50_vol_wk_v1"
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
    ema_50 = np.full(len(close_1d), np.nan)
    
    if len(close_1d) >= 50:
        multiplier = 2 / (50 + 1)
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] - ema_50[i-1]) * multiplier + ema_50[i-1]
    
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 1-week volume average for confirmation
    df_1w = get_htf_data(prices, '1w')
    vol_1w = df_1w['volume'].values
    vol_avg_1w = np.full(len(vol_1w), np.nan)
    
    for i in range(4, len(vol_1w)):  # 5-period average
        vol_avg_1w[i] = np.mean(vol_1w[i-4:i+1])
    
    vol_avg_aligned = align_htf_to_ltf(prices, df_1w, vol_avg_1w)
    
    # Donchian channels (20-period) on 4h data
    donchian_high = np.full(n, np.nan)
    donchian_low = np.full(n, np.nan)
    
    for i in range(19, n):
        donchian_high[i] = np.max(high[i-19:i+1])
        donchian_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 19, 4)  # EMA needs 50, Donchian needs 19, volume needs 4
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(donchian_high[i]) or 
            np.isnan(donchian_low[i]) or np.isnan(vol_avg_aligned[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x weekly average
        volume_filter = volume[i] > vol_avg_aligned[i] * 1.5
        
        # Trend filter: price above/below EMA50
        uptrend = close[i] > ema_50_aligned[i]
        downtrend = close[i] < ema_50_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price breaks below Donchian low or stoploss
            if (close[i] < donchian_low[i] or 
                close[i] < entry_price - 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price breaks above Donchian high or stoploss
            if (close[i] > donchian_high[i] or 
                close[i] > entry_price + 2.5 * np.abs(high[i] - low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakout with volume and trend alignment
            if volume_filter:
                # Long: break above Donchian high in uptrend
                if close[i] > donchian_high[i] and uptrend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: break below Donchian low in downtrend
                elif close[i] < donchian_low[i] and downtrend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals