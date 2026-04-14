#!/usr/bin/env python3
# 4h_1d_OrderBlock_Strategy_v1
# Hypothesis: On 4h timeframe, use 1-day order blocks (unmitigated supply/demand zones) with volume confirmation and RSI filter.
# Identifies institutional order blocks where price moved strongly away with high volume, then returns to test the zone.
# Long when price retraces to bullish order block with RSI < 40 and volume > 1.5x average.
# Short when price retraces to bearish order block with RSI > 60 and volume > 1.5x average.
# Exit when price moves to the opposite order block or RSI reaches opposite extreme.
# Designed to work in both bull and bear markets by following institutional footprints.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Load 1d data for order block identification
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Identify bullish and bearish order blocks on 1d
    # Bullish OB: strong down candle followed by strong up candle that breaks the down candle's high
    # Bearish OB: strong up candle followed by strong down candle that breaks the up candle's low
    bullish_ob_top = np.full(len(df_1d), np.nan)
    bullish_ob_bottom = np.full(len(df_1d), np.nan)
    bearish_ob_top = np.full(len(df_1d), np.nan)
    bearish_ob_bottom = np.full(len(df_1d), np.nan)
    
    for i in range(2, len(df_1d)):
        if np.isnan(high_1d[i]) or np.isnan(low_1d[i]) or np.isnan(close_1d[i]) or \
           np.isnan(high_1d[i-1]) or np.isnan(low_1d[i-1]) or np.isnan(close_1d[i-1]) or \
           np.isnan(high_1d[i-2]) or np.isnan(low_1d[i-2]) or np.isnan(close_1d[i-2]):
            continue
            
        # Bullish OB: bearish candle (i-2) followed by bullish candle (i-1) that breaks (i-2) high
        if close_1d[i-2] < open_1d[i-2] and close_1d[i-1] > open_1d[i-1] and high_1d[i-1] > high_1d[i-2]:
            bullish_ob_top[i-1] = high_1d[i-2]  # Top of OB is the high of the bearish candle
            bullish_ob_bottom[i-1] = low_1d[i-1]  # Bottom is the low of the bullish candle
            
        # Bearish OB: bullish candle (i-2) followed by bearish candle (i-1) that breaks (i-2) low
        if close_1d[i-2] > open_1d[i-2] and close_1d[i-1] < open_1d[i-1] and low_1d[i-1] < low_1d[i-2]:
            bearish_ob_top[i-1] = high_1d[i-1]  # Top is the high of the bearish candle
            bearish_ob_bottom[i-1] = low_1d[i-2]  # Bottom is the low of the bullish candle
    
    # Forward fill the order block levels until they are mitigated
    for i in range(1, len(bullish_ob_top)):
        if not np.isnan(bullish_ob_top[i]):
            bullish_ob_top[i] = bullish_ob_top[i]
            bullish_ob_bottom[i] = bullish_ob_bottom[i]
        else:
            bullish_ob_top[i] = bullish_ob_top[i-1]
            bullish_ob_bottom[i] = bullish_ob_bottom[i-1]
            
        if not np.isnan(bearish_ob_top[i]):
            bearish_ob_top[i] = bearish_ob_top[i]
            bearish_ob_bottom[i] = bearish_ob_bottom[i]
        else:
            bearish_ob_top[i] = bearish_ob_top[i-1]
            bearish_ob_bottom[i] = bearish_ob_bottom[i-1]
    
    # Mitigate order blocks when price breaks through them
    for i in range(len(df_1d)):
        if not np.isnan(bullish_ob_top[i]) and high_1d[i] > bullish_ob_top[i]:
            bullish_ob_top[i] = np.nan
            bullish_ob_bottom[i] = np.nan
        if not np.isnan(bearish_ob_bottom[i]) and low_1d[i] < bearish_ob_bottom[i]:
            bearish_ob_top[i] = np.nan
            bearish_ob_bottom[i] = np.nan
    
    # Align order block levels to 4h timeframe
    bullish_ob_top_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_top)
    bullish_ob_bottom_aligned = align_htf_to_ltf(prices, df_1d, bullish_ob_bottom)
    bearish_ob_top_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_top)
    bearish_ob_bottom_aligned = align_htf_to_ltf(prices, df_1d, bearish_ob_bottom)
    
    # Calculate RSI on 1d
    delta = np.diff(close_1d, prepend=close_1d[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    rs = np.zeros_like(close_1d)
    rsi = np.zeros_like(close_1d)
    
    avg_gain[13] = np.mean(gain[1:14])
    avg_loss[13] = np.mean(loss[1:14])
    
    for i in range(14, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
        if avg_loss[i] != 0:
            rs[i] = avg_gain[i] / avg_loss[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        else:
            rsi[i] = 100
    
    # Align RSI to 4h timeframe
    rsi_aligned = align_htf_to_ltf(prices, df_1d, rsi)
    
    # Calculate volume moving average
    vol_ma_20 = np.full_like(volume, np.nan)
    for i in range(19, len(volume)):
        vol_ma_20[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0
    position_size = 0.25  # 25% position
    
    for i in range(20, n):
        # Skip if critical data is NaN
        if (np.isnan(bullish_ob_top_aligned[i]) or np.isnan(bullish_ob_bottom_aligned[i]) or
            np.isnan(bearish_ob_top_aligned[i]) or np.isnan(bearish_ob_bottom_aligned[i]) or
            np.isnan(rsi_aligned[i]) or np.isnan(vol_ma_20[i]) or vol_ma_20[i] <= 0):
            signals[i] = 0.0
            continue
        
        volume_ratio = volume[i] / vol_ma_20[i]
        
        if position == 0:
            # Long: price at bullish OB, RSI oversold, volume confirmation
            if (low[i] <= bullish_ob_top_aligned[i] and 
                high[i] >= bullish_ob_bottom_aligned[i] and
                rsi_aligned[i] < 40 and
                volume_ratio > 1.5):
                position = 1
                signals[i] = position_size
            # Short: price at bearish OB, RSI overbought, volume confirmation
            elif (low[i] <= bearish_ob_top_aligned[i] and 
                  high[i] >= bearish_ob_bottom_aligned[i] and
                  rsi_aligned[i] > 60 and
                  volume_ratio > 1.5):
                position = -1
                signals[i] = -position_size
            else:
                signals[i] = 0.0
        elif position == 1:
            # Exit long: price reaches bearish OB or RSI overbought
            if (low[i] <= bearish_ob_top_aligned[i] and 
                high[i] >= bearish_ob_bottom_aligned[i]) or \
               rsi_aligned[i] > 60:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = position_size
        elif position == -1:
            # Exit short: price reaches bullish OB or RSI oversold
            if (low[i] <= bullish_ob_top_aligned[i] and 
                high[i] >= bullish_ob_bottom_aligned[i]) or \
               rsi_aligned[i] < 40:
                position = 0
                signals[i] = 0.0
            else:
                signals[i] = -position_size
    
    return signals

name = "4h_1d_OrderBlock_Strategy_v1"
timeframe = "4h"
leverage = 1.0