#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Camarilla pivot reversal with 1-day trend filter.
# Uses daily pivot levels (R3/S3 for mean reversion, R4/S4 for breakout) to identify
# reversal and continuation opportunities. 1-day EMA200 establishes trend bias:
# - In uptrend: look for mean reversion at S3/S4, breakout above R4
# - In downtrend: look for mean reversion at R3/R4, breakdown below S4
# Volume confirmation filters false signals. Designed for 6h timeframe to target
# 50-150 trades over 4 years with precise pivot-based entries.

name = "6h_camarilla_pivot_rev_trend_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
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
    
    # Align EMA200 to 6h timeframe (shifted by 1 day for no look-ahead)
    ema_200_aligned = align_htf_to_ltf(prices, df_1d, ema_200_1d)
    
    # Calculate daily pivot points (using previous day's OHLC)
    # Pivot = (H + L + C) / 3
    # R4 = C + ((H-L) * 1.1/2), S4 = C - ((H-L) * 1.1/2)
    # R3 = C + ((H-L) * 1.1/4), S3 = C - ((H-L) * 1.1/4)
    # R2 = C + ((H-L) * 1.1/6), S2 = C - ((H-L) * 1.1/6)
    # R1 = C + ((H-L) * 1.1/12), S1 = C - ((H-L) * 1.1/12)
    
    # Shift by 1 to use previous day's data (no look-ahead)
    if len(close_1d) >= 2:
        prev_high = np.roll(high, 1)[::4][1:]  # Assuming 4 6h bars per day, simplified
        prev_low = np.roll(low, 1)[::4][1:]
        prev_close = np.roll(close_1d, 1)[1:]
    else:
        prev_high = np.zeros_like(close_1d)
        prev_low = np.zeros_like(close_1d)
        prev_close = np.zeros_like(close_1d)
    
    # Calculate pivot levels for each day
    pivot = np.full(len(close_1d), np.nan)
    r4 = np.full(len(close_1d), np.nan)
    s4 = np.full(len(close_1d), np.nan)
    r3 = np.full(len(close_1d), np.nan)
    s3 = np.full(len(close_1d), np.nan)
    
    valid_idx = (~np.isnan(prev_high)) & (~np.isnan(prev_low)) & (~np.isnan(prev_close))
    if np.any(valid_idx):
        pivot[valid_idx] = (prev_high[valid_idx] + prev_low[valid_idx] + prev_close[valid_idx]) / 3.0
        rng = prev_high[valid_idx] - prev_low[valid_idx]
        r4[valid_idx] = prev_close[valid_idx] + (rng * 1.1 / 2)
        s4[valid_idx] = prev_close[valid_idx] - (rng * 1.1 / 2)
        r3[valid_idx] = prev_close[valid_idx] + (rng * 1.1 / 4)
        s3[valid_idx] = prev_close[valid_idx] - (rng * 1.1 / 4)
    
    # Align pivot levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, pivot)
    r4_aligned = align_htf_to_ltf(prices, df_1d, r4)
    s4_aligned = align_htf_to_ltf(prices, df_1d, s4)
    r3_aligned = align_htf_to_ltf(prices, df_1d, r3)
    s3_aligned = align_htf_to_ltf(prices, df_1d, s3)
    
    # Volume confirmation: 6h volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if required data not available
        if (np.isnan(ema_200_aligned[i]) or np.isnan(pivot_aligned[i]) or 
            np.isnan(r3_aligned[i]) or np.isnan(s3_aligned[i]) or
            np.isnan(r4_aligned[i]) or np.isnan(s4_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Trend bias
        bullish_trend = close[i] > ema_200_aligned[i]
        bearish_trend = close[i] < ema_200_aligned[i]
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: reverse signal or stoploss (2x ATR approximation)
            atr_approx = (r4_aligned[i] - s4_aligned[i]) / 6  # Rough ATR estimate from pivot width
            if atr_approx <= 0:
                atr_approx = 0.01  # Fallback
            stop_loss = entry_price - 2.0 * atr_approx
            
            # Reverse conditions: price breaks S3 in uptrend or breaks S4 in any trend
            if (close[i] < s3_aligned[i] and bearish_trend) or close[i] < s4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: reverse signal or stoploss
            atr_approx = (r4_aligned[i] - s4_aligned[i]) / 6
            if atr_approx <= 0:
                atr_approx = 0.01
            stop_loss = entry_price + 2.0 * atr_approx
            
            # Reverse conditions: price breaks R3 in downtrend or breaks R4 in any trend
            if (close[i] > r3_aligned[i] and bullish_trend) or close[i] > r4_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries
            if volume_filter:
                # Long entries: mean reversion at S3/S4 in uptrend, breakout above R4
                if bullish_trend:
                    if close[i] <= s3_aligned[i] * 1.001 and close[i] >= s3_aligned[i] * 0.999:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    elif close[i] <= s4_aligned[i] * 1.001 and close[i] >= s4_aligned[i] * 0.999:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                    elif close[i] > r4_aligned[i] and close[i-1] <= r4_aligned[i-1]:
                        signals[i] = 0.25
                        position = 1
                        entry_price = close[i]
                # Short entries: mean reversion at R3/R4 in downtrend, breakdown below S4
                elif bearish_trend:
                    if close[i] >= r3_aligned[i] * 0.999 and close[i] <= r3_aligned[i] * 1.001:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
                    elif close[i] >= r4_aligned[i] * 0.999 and close[i] <= r4_aligned[i] * 1.001:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
                    elif close[i] < s4_aligned[i] and close[i-1] >= s4_aligned[i-1]:
                        signals[i] = -0.25
                        position = -1
                        entry_price = close[i]
            else:
                signals[i] = 0.0
    
    return signals