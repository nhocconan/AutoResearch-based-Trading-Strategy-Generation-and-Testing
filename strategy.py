#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6h Donchian(20) breakout + daily RSI(14) filter + volume confirmation
# Donchian breakouts capture momentum with defined risk. Daily RSI filters for 
# overbought/oversold conditions to avoid false breakouts. Volume confirms 
# institutional participation. Designed for 6h timeframe to target 50-150 trades over 4 years.
# Works in bull markets via breakout continuation and bear via faded false breaks.

name = "6h_donchian20_1d_rsi_vol_v1"
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
    
    # 1-day RSI(14) for overbought/oversold filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # RSI calculation
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.full(len(close_1d), np.nan)
    avg_loss = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        if i < 14:
            avg_gain[i] = np.mean(gain[1:i+1]) if i > 0 else 0
            avg_loss[i] = np.mean(loss[1:i+1]) if i > 0 else 0
        else:
            avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rsi = np.full(len(close_1d), np.nan)
    for i in range(14, len(close_1d)):
        if avg_loss[i] != 0:
            rs = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs))
        else:
            rsi[i] = 100
    
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # 6-day volume average for confirmation
    vol_ma_6d = np.full(n, np.nan)
    for i in range(5, n):  # 6-period average
        vol_ma_6d[i] = np.mean(volume[i-5:i+1])
    
    # Donchian channels (20-period)
    highest_high = np.full(n, np.nan)
    lowest_low = np.full(n, np.nan)
    
    for i in range(19, n):
        highest_high[i] = np.max(high[i-19:i+1])
        lowest_low[i] = np.min(low[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(19, 14, 5)  # Donchian needs 19, RSI needs 14, volume needs 5
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or 
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_6d[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition: current volume > 1.2x 6-day average
        volume_filter = volume[i] > vol_ma_6d[i] * 1.2
        
        # RSI conditions
        rsi_overbought = rsi_aligned[i] > 70
        rsi_oversold = rsi_aligned[i] < 30
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: RSI overbought or price breaks below lower Donchian
            if (rsi_overbought or close[i] < lowest_low[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: RSI oversold or price breaks above upper Donchian
            if (rsi_oversold or close[i] > highest_high[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries: breakouts with volume and RSI filter
            if volume_filter:
                # Long: price breaks above upper Donchian and not overbought
                if close[i] > highest_high[i] and not rsi_overbought:
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: price breaks below lower Donchian and not oversold
                elif close[i] < lowest_low[i] and not rsi_oversold:
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals