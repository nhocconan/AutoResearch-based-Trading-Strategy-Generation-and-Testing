#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4-hour Donchian(20) breakout with 12-hour EMA trend filter and volume confirmation.
# Uses 12-hour EMA(20) for trend direction and 12-hour volume confirmation to filter breakouts.
# Designed for 4h timeframe to target 75-200 trades over 4 years with balanced frequency.
# Works in bull markets via upward breakouts and bear markets via downward breakouts with trend alignment.

name = "4h_donchian20_12h_ema_vol_v1"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 12-hour EMA(20) for trend direction
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate EMA(20) on 12h closes
    ema_12h = np.full(len(close_12h), np.nan)
    if len(close_12h) >= 20:
        ema_12h[19] = np.mean(close_12h[:20])
        for i in range(20, len(close_12h)):
            ema_12h[i] = (close_12h[i] * 2 / (20 + 1)) + (ema_12h[i-1] * (19 / (20 + 1)))
    
    # Calculate 12h volume moving average (20-period)
    vol_ma_12h = np.full(len(volume_12h), np.nan)
    if len(volume_12h) >= 20:
        for i in range(19, len(volume_12h)):
            vol_ma_12h[i] = np.mean(volume_12h[i-19:i+1])
    
    # Align 12h indicators to 4h timeframe
    ema_12h_aligned = align_htf_to_ltf(prices, df_12h, ema_12h)
    vol_ma_12h_aligned = align_htf_to_ltf(prices, df_12h, vol_ma_12h)
    
    # 4-hour Donchian channel (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    # Volume confirmation: 4h volume > 1.3x 20-period average
    vol_ma_4h = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma_4h[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_12h_aligned[i]) or np.isnan(vol_ma_12h_aligned[i]) or 
            np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(vol_ma_4h[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume conditions
        vol_4h_filter = volume[i] > vol_ma_4h[i] * 1.3
        vol_12h_filter = volume_12h_aligned[i] > vol_ma_12h_aligned[i] * 1.3 if not np.isnan(volume_12h_aligned[i]) else False
        
        # Trend filter: price above/below 12h EMA
        uptrend = close[i] > ema_12h_aligned[i]
        downtrend = close[i] < ema_12h_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: trend reversal or stoploss
            if (not uptrend or 
                close[i] < entry_price - 2.0 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend reversal or stoploss
            if (not downtrend or 
                close[i] > entry_price + 2.0 * (highest_high[i] - lowest_low[i])):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend confirmation
            if vol_4h_filter and vol_12h_filter:
                # Long: breakout above resistance in uptrend
                if (highest_high[i] > highest_high[i-1] and 
                    close[i] > highest_high[i-1] and uptrend):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: breakdown below support in downtrend
                elif (lowest_low[i] < lowest_low[i-1] and 
                      close[i] < lowest_low[i-1] and downtrend):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals