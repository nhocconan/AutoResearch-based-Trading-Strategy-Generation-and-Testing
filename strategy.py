#!/usr/bin/env python3
# 6h_1w_1d_rsi_pivot_reversion_v1
# Hypothesis: 6-hour mean-reversion strategy using weekly RSI trend filter and daily pivot points.
# In strong weekly trends (RSI > 50), we buy dips to S1/S2; in weak weekly trends (RSI < 50), we sell rallies to R1/R2.
# Uses weekly RSI(14) to determine regime and daily pivot points for entry/exit levels.
# Designed for low trade frequency (15-35/year) to minimize fee drag in 6h timeframe.
# Works in bull markets by buying dips in uptrends and in bear markets by selling rallies in downtrends.

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for RSI trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    # Get daily data for pivot points
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    # Calculate weekly RSI(14)
    close_1w = df_1w['close'].values
    delta = np.diff(close_1w, prepend=close_1w[0])
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    
    # Wilder's smoothing
    avg_gain = np.zeros_like(gain)
    avg_loss = np.zeros_like(loss)
    avg_gain[14] = np.mean(gain[1:15])  # first average
    avg_loss[14] = np.mean(loss[1:15])
    
    for i in range(15, len(gain)):
        avg_gain[i] = (avg_gain[i-1] * 13 + gain[i]) / 14
        avg_loss[i] = (avg_loss[i-1] * 13 + loss[i]) / 14
    
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    rsi[:14] = np.nan  # not enough data
    
    # Weekly trend filter: RSI > 50 = bullish regime, RSI < 50 = bearish regime
    weekly_bullish_regime = rsi > 50
    weekly_bearish_regime = rsi < 50
    
    # Align weekly RSI regime to 6h timeframe
    weekly_bullish_6h = align_htf_to_ltf(prices, df_1w, weekly_bullish_regime.astype(float))
    weekly_bearish_6h = align_htf_to_ltf(prices, df_1w, weekly_bearish_regime.astype(float))
    
    # Calculate daily pivot points using PREVIOUS day's data (no look-ahead)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Previous day's values for pivot calculation
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    prev_close = np.roll(close_1d, 1)
    # Set first day's previous values to NaN (no data yet)
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_close[0] = np.nan
    
    # Pivot point calculation
    pivot = (prev_high + prev_low + prev_close) / 3
    range_val = prev_high - prev_low
    
    # Support and resistance levels
    s1 = 2 * pivot - prev_high
    s2 = pivot - range_val
    r1 = 2 * pivot - prev_low
    r2 = pivot + range_val
    
    # Align pivot levels to 6h timeframe
    s1_6h = align_htf_to_ltf(prices, df_1d, s1)
    s2_6h = align_htf_to_ltf(prices, df_1d, s2)
    r1_6h = align_htf_to_ltf(prices, df_1d, r1)
    r2_6h = align_htf_to_ltf(prices, df_1d, r2)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if data not ready
        if (np.isnan(weekly_bullish_6h[i]) or np.isnan(weekly_bearish_6h[i]) or 
            np.isnan(s1_6h[i]) or np.isnan(s2_6h[i]) or np.isnan(r1_6h[i]) or np.isnan(r2_6h[i])):
            signals[i] = 0.0
            continue
        
        # Entry conditions: mean reversion at pivot levels with weekly regime alignment
        # In bullish weekly regime: buy near S1/S2
        long_entry = ((close[i] <= s1_6h[i]) or (close[i] <= s2_6h[i])) and weekly_bullish_6h[i] == 1.0
        # In bearish weekly regime: sell near R1/R2
        short_entry = ((close[i] >= r1_6h[i]) or (close[i] >= r2_6h[i])) and weekly_bearish_6h[i] == 1.0
        
        # Exit conditions: return to pivot or weekly regime change
        long_exit = (close[i] >= pivot[i]) or (weekly_bullish_6h[i] == 0.0 and weekly_bearish_6h[i] == 0.0)
        short_exit = (close[i] <= pivot[i]) or (weekly_bullish_6h[i] == 0.0 and weekly_bearish_6h[i] == 0.0)
        
        if long_entry and position != 1:
            position = 1
            signals[i] = 0.25
        elif short_entry and position != -1:
            position = -1
            signals[i] = -0.25
        elif position == 1 and long_exit:
            position = 0
            signals[i] = 0.0
        elif position == -1 and short_exit:
            position = 0
            signals[i] = 0.0
        else:
            # Hold current position
            if position == 1:
                signals[i] = 0.25
            elif position == -1:
                signals[i] = -0.25
            else:
                signals[i] = 0.0
    
    return signals

name = "6h_1w_1d_rsi_pivot_reversion_v1"
timeframe = "6h"
leverage = 1.0