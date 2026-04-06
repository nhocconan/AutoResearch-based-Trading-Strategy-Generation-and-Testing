#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Donchian(20) breakout with 1-day trend filter and volume confirmation.
# Uses 1-day EMA(50) to determine trend direction: long when price > EMA50, short when price < EMA50.
# Entry on 6h Donchian breakout in direction of 1d trend, with volume > 1.5x 20-period average.
# Designed for 6h timeframe to target 50-150 trades over 4 years with moderate frequency.
# Works in bull markets via trend-following breakouts and in bear via short breakdowns.

name = "6h_donchian20_1d_trend_vol_v1"
timeframe = "6h"
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
    
    # Calculate EMA(50) on daily data
    ema_50 = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema_50[49] = np.mean(close_1d[:50])  # Simple average for first value
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] * 2/51) + (ema_50[i-1] * (1 - 2/51))
    
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 6-day volume average for confirmation (20 periods on 6h ≈ 5 days)
    vol_ma = np.full(len(volume), np.nan)
    for i in range(19, len(volume)):  # 20-period moving average
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(50, 20)  # EMA needs 50, Donchian needs 20
    
    for i in range(start, n):
        # Skip if required data not available
        if np.isnan(ema_50_aligned[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Calculate 6h Donchian channels (20-period)
        highest_high = np.max(high[i-19:i+1])
        lowest_low = np.min(low[i-19:i+1])
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Trend condition from 1d EMA50
        uptrend = close > ema_50_aligned[i]
        downtrend = close < ema_50_aligned[i]
        
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
            # Look for entries: breakout in direction of 1d trend with volume
            if volume_filter:
                # Long: breakout above upper band in uptrend
                if close[i] > highest_high and uptrend:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below lower band in downtrend
                elif close[i] < lowest_low and downtrend:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals