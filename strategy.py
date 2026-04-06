#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot reversal from daily pivots with volume confirmation.
# Uses daily Camarilla levels (R3/S3 for reversals, R4/S4 for breakouts) to capture mean reversion in ranges
# and breakout momentum in trends. Volume filter reduces false signals. Designed for 50-150 trades over 4 years.

name = "6h_camarilla1d_vol_v1"
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
    
    # 14-period ATR for stops
    atr = np.full(n, np.nan)
    if n >= 14:
        tr = np.maximum(
            high[1:] - low[1:],
            np.abs(high[1:] - close[:-1]),
            np.abs(low[1:] - close[:-1])
        )
        if len(tr) > 0:
            atr[14] = np.mean(tr[:14])
            for i in range(15, n):
                atr[i] = (atr[i-1] * 13 + tr[i-1]) / 14
    
    # Daily OHLC for Camarilla calculation
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for previous day
    camarilla_r4 = np.full(len(close_1d), np.nan)
    camarilla_r3 = np.full(len(close_1d), np.nan)
    camarilla_s3 = np.full(len(close_1d), np.nan)
    camarilla_s4 = np.full(len(close_1d), np.nan)
    
    for i in range(1, len(close_1d)):
        # Previous day's range
        prev_high = high_1d[i-1]
        prev_low = low_1d[i-1]
        prev_close = close_1d[i-1]
        range_val = prev_high - prev_low
        
        if not np.isnan(range_val) and range_val > 0:
            camarilla_r4[i] = prev_close + range_val * 1.1 / 2
            camarilla_r3[i] = prev_close + range_val * 1.1 / 4
            camarilla_s3[i] = prev_close - range_val * 1.1 / 4
            camarilla_s4[i] = prev_close - range_val * 1.1 / 2
    
    # Align Camarilla levels to 6h timeframe (shifted by 1 day for lookback)
    r4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s3)
    s4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s4)
    
    # Volume filter: current volume > 1.3x average over last 20 periods
    vol_ma = np.full(n, np.nan)
    for i in range(20, n):
        vol_ma[i] = np.mean(volume[i-20:i])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    # Start from warmup period
    start = max(20, 14)
    
    for i in range(start, n):
        # Skip if required data not available
        if (np.isnan(atr[i]) or np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: price reaches S3 (mean reversion target) or stoploss hit
            if (close[i] <= s3_aligned[i] or
                close[i] < entry_price - 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: price reaches R3 (mean reversion target) or stoploss hit
            if (close[i] >= r3_aligned[i] or
                close[i] > entry_price + 2.5 * atr[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            # Long reversal: price touches/below S4 and rebounds with volume (mean reversion in range)
            if (close[i] <= s4_aligned[i] and volume_filter and 
                close[i] > open[i]):  # bullish candle
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short reversal: price touches/above R4 and rejects with volume (mean reversion in range)
            elif (close[i] >= r4_aligned[i] and volume_filter and 
                  close[i] < open[i]):  # bearish candle
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            # Long breakout: price breaks above R4 with volume (trend continuation)
            elif (close[i] > r4_aligned[i] and volume_filter and 
                  close[i] > open[i]):  # bullish breakout
                signals[i] = 0.25
                position = 1
                entry_price = close[i]
            # Short breakout: price breaks below S4 with volume (trend continuation)
            elif (close[i] < s4_aligned[i] and volume_filter and 
                  close[i] < open[i]):  # bearish breakout
                signals[i] = -0.25
                position = -1
                entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals