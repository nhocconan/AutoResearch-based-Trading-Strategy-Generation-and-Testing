#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12-hour Donchian(10) breakout with daily trend filter and volume confirmation.
# Uses 1-day EMA(50) to establish trend direction (long above, short below).
# Breakouts in direction of daily trend with volume > 2x average capture strong moves.
# Designed for 12h timeframe to target 50-150 trades over 4 years with low frequency.
# Works in bull/bear markets via trend-following logic with volatility filter.

name = "12h_donchian10_1d_ema_vol_v1"
timeframe = "12h"
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
        multiplier = 2 / (50 + 1)
        ema_50[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema_50[i] = (close_1d[i] - ema_50[i-1]) * multiplier + ema_50[i-1]
    
    # Align EMA to 12h timeframe (shifted by 1 day for no look-ahead)
    ema_50_aligned = align_htf_to_ltf(prices, df_1d, ema_50)
    
    # 12-hour Donchian channel (10-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(9, n):
        highest_high[i] = np.max(high[i-9:i+1])
        lowest_low[i] = np.min(low[i-9:i+1])
    
    # Volume confirmation: 12h volume > 2x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_50_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 2x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 2.0
        
        # Trend filter: price above/below daily EMA(50)
        bullish_trend = close[i] > ema_50_aligned[i]
        bearish_trend = close[i] < ema_50_aligned[i]
        
        # Check exits and stoploss (2x ATR approximation using Donchian width)
        if position == 1:  # long position
            # Exit: trend reversal or stoploss
            if (not bullish_trend or 
                close[i] < entry_price - 2.0 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend reversal or stoploss
            if (not bearish_trend or 
                close[i] > entry_price + 2.0 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of trend with volume confirmation
            if volume_filter:
                # Long: breakout above resistance with bullish trend
                if (highest_high[i] > highest_high[i-1] and 
                    close[i] > highest_high[i] and bullish_trend):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below support with bearish trend
                elif (lowest_low[i] < lowest_low[i-1] and 
                      close[i] < lowest_low[i] and bearish_trend):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals