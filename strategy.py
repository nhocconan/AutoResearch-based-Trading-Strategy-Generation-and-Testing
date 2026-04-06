#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1-hour Donchian(15) breakout with 4-hour Hull Moving Average (21) trend and 1-day pivot direction.
# Uses 4-hour HMA for trend direction and 1-day pivot for institutional bias.
# Breakouts in direction of both trend and bias with volume confirmation capture institutional moves.
# Designed for 1h timeframe to target 60-150 trades over 4 years (15-37/year) with moderate frequency.
# Works in bull/bear markets via dual trend (HMA) and bias (pivot) filters.

name = "1h_donchian15_hma21_1d_pivot_v1"
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
    
    # 4-hour Hull Moving Average (21) for trend direction
    df_4h = get_htf_data(prices, '4h')
    close_4h = df_4h['close'].values
    
    # WMA function
    def wma(arr, period):
        weights = np.arange(1, period + 1)
        return np.convolve(arr, weights/weights.sum(), mode='same')
    
    # Hull MA: WMA(2 * WMA(n/2) - WMA(n), sqrt(n))
    half_period = 21 // 2
    sqrt_period = int(np.sqrt(21))
    
    wma_half = wma(close_4h, half_period)
    wma_full = wma(close_4h, 21)
    raw_hma = 2 * wma_half - wma_full
    hma_4h = wma(raw_hma, sqrt_period)
    
    # Align HMA to 1h timeframe
    hma_4h_aligned = align_htf_to_ltf(prices, df_4h, hma_4h)
    
    # 1-day pivot points (calculated from prior day)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Pivot point calculations (standard formula)
    pivot = (high_1d + low_1d + close_1d) / 3
    r1 = 2 * pivot - low_1d
    s1 = 2 * pivot - high_1d
    r2 = pivot + (high_1d - low_1d)
    s2 = pivot - (high_1d - low_1d)
    r3 = high_1d + 2 * (pivot - low_1d)
    s3 = low_1d - 2 * (high_1d - pivot)
    
    # Align pivot levels to 1h timeframe (shifted by 1 day for no look-ahead)
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    r2_aligned = align_htf_to_ltf(prices, df_1d, r2)
    s2_aligned = align_htf_to_ltf(prices, df_1d, s2)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # 1-hour Donchian channel (15-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(14, n):
        highest_high[i] = np.max(high[i-14:i+1])
        lowest_low[i] = np.min(low[i-14:i+1])
    
    # Volume confirmation: 1h volume > 1.3x 15-period average
    vol_ma = np.full(n, np.nan)
    for i in range(14, n):
        vol_ma[i] = np.mean(volume[i-14:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(15, n):
        # Skip if required data not available
        if (np.isnan(hma_4h_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r1_aligned[i]) or np.isnan(s1_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.20
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.3x 15-period average
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Trend condition: price above/below HMA
        bullish_trend = close[i] > hma_4h_aligned[i]
        bearish_trend = close[i] < hma_4h_aligned[i]
        
        # Pivot bias: long above pivot, short below pivot
        bullish_bias = close[i] > pivot_aligned[i]
        bearish_bias = close[i] < pivot_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below HMA or pivot or stoploss
            if (close[i] < hma_4h_aligned[i] or 
                close[i] < pivot_aligned[i] or 
                close[i] < entry_price - 2.0 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:  # short position
            # Exit: price above HMA or pivot or stoploss
            if (close[i] > hma_4h_aligned[i] or 
                close[i] > pivot_aligned[i] or 
                close[i] > entry_price + 2.0 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
        else:
            # Look for entries in direction of both trend and bias
            if volume_filter:
                # Long: breakout above resistance with bullish trend and bias
                if (highest_high[i] > r1_aligned[i] and 
                    close[i] > highest_high[i-1] and bullish_trend and bullish_bias):
                    signals[i] = 0.20
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below support with bearish trend and bias
                elif (lowest_low[i] < s1_aligned[i] and 
                      close[i] < lowest_low[i-1] and bearish_trend and bearish_bias):
                    signals[i] = -0.20
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals