#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 1-day EMA200 trend filter and volume confirmation.
# Uses 1-day EMA200 for strong trend bias (long above EMA200, short below EMA200).
# Breakouts in direction of EMA200 trend with volume > 1.5x average capture institutional moves.
# EMA200 provides robust trend filter that works in both bull and bear markets.
# Designed for 4h timeframe to target 75-200 trades over 4 years with proven structure.

name = "4h_donchian20_1d_ema200_vol_v1"
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
    
    # 1-day EMA200 for trend bias
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily closes
    ema_200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema_200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema_200_1d[i] = (close_1d[i] * 2 / 201) + (ema_200_1d[i-1] * 199 / 201)
    
    # Align EMA200 to 4h timeframe (shifted by 1 day for no look-ahead)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # 4-hour Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 4h volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(highest_high[i]) or 
            np.isnan(lowest_low[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.5x 20-period average
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Trend bias: long above EMA200, short below EMA200
        bullish_bias = close[i] > ema_200_aligned[i]
        bearish_bias = close[i] < ema_200_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price below EMA200 or stoploss (2x ATR approximation using Donchian width)
            donch_width = highest_high[i] - lowest_low[i]
            if donch_width > 0:
                stop_loss_level = entry_price - 2.0 * donch_width
            else:
                stop_loss_level = entry_price - 2.0 * (highest_high[i] - lowest_low[i] + 0.001)
            
            if (close[i] < ema_200_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price above EMA200 or stoploss
            donch_width = highest_high[i] - lowest_low[i]
            if donch_width > 0:
                stop_loss_level = entry_price + 2.0 * donch_width
            else:
                stop_loss_level = entry_price + 2.0 * (highest_high[i] - lowest_low[i] + 0.001)
            
            if (close[i] > ema_200_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries in direction of EMA200 trend
            if volume_filter:
                # Long: breakout above resistance with bullish bias
                if (highest_high[i] > highest_high[i-1] and 
                    close[i] > highest_high[i-1] and bullish_bias):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below support with bearish bias
                elif (lowest_low[i] < lowest_low[i-1] and 
                      close[i] < lowest_low[i-1] and bearish_bias):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals