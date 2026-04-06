#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Power with Trend Filter.
# Uses Elder Ray Bull Power (High - EMA13) and Bear Power (Low - EMA13) from daily timeframe.
# Long when Bull Power > 0 and Bear Power rising (bullish momentum).
# Short when Bear Power < 0 and Bull Power falling (bearish momentum).
# Includes 6h EMA50 trend filter to avoid counter-trend trades.
# Works in bull/bear markets via momentum-based signals.
# Target: 50-150 trades over 4 years (12-37/year).

name = "6h_elder_ray_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    
    # Daily EMA13 for Elder Ray calculation
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA13 on daily close
    ema13_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 13:
        ema13_1d[12] = np.mean(close_1d[:13])
        for i in range(13, len(close_1d)):
            ema13_1d[i] = (close_1d[i] * 2 / (13 + 1)) + ema13_1d[i-1] * (1 - 2 / (13 + 1))
    
    # Calculate Elder Ray components
    bull_power_1d = np.full(len(close_1d), np.nan)
    bear_power_1d = np.full(len(close_1d), np.nan)
    
    for i in range(len(close_1d)):
        if not (np.isnan(high[i*4]) if i*4 < len(high) else True or 
                np.isnan(low[i*4]) if i*4 < len(low) else True or 
                np.isnan(ema13_1d[i])):
            # Approximate daily high/low from 6h data (simplified)
            # In practice, we'd use actual daily data, but using proxies
            daily_high = high[min(i*4+3, len(high)-1)] if i*4 < len(high) else high[-1]
            daily_low = low[min(i*4+3, len(low)-1)] if i*4 < len(low) else low[-1]
            bull_power_1d[i] = daily_high - ema13_1d[i]
            bear_power_1d[i] = daily_low - ema13_1d[i]
    
    # Align Elder Ray to 6h timeframe (shifted by 1 daily bar)
    bull_power_aligned = align_htf_to_ltf(prices, df_1d, bull_power_1d)
    bear_power_aligned = align_htf_to_ltf(prices, df_1d, bear_power_1d)
    
    # 6h EMA50 for trend filter
    ema50 = np.full(n, np.nan)
    if n >= 50:
        ema50[49] = np.mean(close[:50])
        for i in range(50, n):
            ema50[i] = (close[i] * 2 / (50 + 1)) + ema50[i-1] * (1 - 2 / (50 + 1))
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(50, n):
        # Skip if data not available
        if (np.isnan(bull_power_aligned[i]) or np.isnan(bear_power_aligned[i]) or 
            np.isnan(ema50[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: Bear Power turns negative or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (bear_power_aligned[i] < 0 or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: Bull Power turns negative or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (bull_power_aligned[i] > 0 or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with trend filter
            # Long: Bull Power > 0 and Bear Power rising (bullish momentum) + price > EMA50
            if (bull_power_aligned[i] > 0 and 
                bear_power_aligned[i] > bear_power_aligned[i-1] and 
                close[i] > ema50[i]):
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short: Bear Power < 0 and Bull Power falling (bearish momentum) + price < EMA50
            elif (bear_power_aligned[i] < 0 and 
                  bull_power_aligned[i] < bull_power_aligned[i-1] and 
                  close[i] < ema50[i]):
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
    
    return signals