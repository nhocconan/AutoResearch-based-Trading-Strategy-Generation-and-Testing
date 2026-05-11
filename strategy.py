#!/usr/bin/env python3
# 1d_1w_WeeklyBreakout_KAMA_TrendFilter
# Hypothesis: Daily price breaks above/below weekly high/low with KAMA trend filter and volume confirmation.
# Uses KAMA's adaptive smoothing to catch trends while avoiding whipsaws in ranging markets.
# Designed for low trade frequency (10-25/year) to minimize fee drag on 1d timeframe.

name = "1d_1w_WeeklyBreakout_KAMA_TrendFilter"
timeframe = "1d"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get weekly data for trend and reference levels
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 2:
        return np.zeros(n)
    
    # Daily OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Weekly high and low
    weekly_high = df_1w['high'].values
    weekly_low = df_1w['low'].values
    
    # Shift by 1 to use previous week's data (no look-ahead)
    weekly_high_prev = np.roll(weekly_high, 1)
    weekly_low_prev = np.roll(weekly_low, 1)
    weekly_high_prev[0] = weekly_high[0]
    weekly_low_prev[0] = weekly_low[0]
    
    # Align weekly levels to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high_prev)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low_prev)
    
    # KAMA trend filter on weekly close
    weekly_close = df_1w['close'].values
    # Calculate Efficiency Ratio (ER) over 10 periods
    change = np.abs(np.diff(weekly_close, 10))
    volatility = np.sum(np.abs(np.diff(weekly_close, 1)), axis=0) if len(weekly_close) > 1 else np.zeros_like(weekly_close)
    # Handle first 10 values
    change = np.concatenate([np.full(10, change[10] if len(change) > 10 else 0), change])
    volatility = np.concatenate([np.full(10, volatility[10] if len(volatility) > 10 else 0), volatility])
    # Avoid division by zero
    er = np.where(volatility != 0, change / volatility, 0)
    # Smoothing constants
    sc = (er * (2/(2+1) - 2/(30+1)) + 2/(30+1)) ** 2
    # Calculate KAMA
    kama = np.zeros_like(weekly_close)
    kama[0] = weekly_close[0]
    for i in range(1, len(weekly_close)):
        kama[i] = kama[i-1] + sc[i] * (weekly_close[i] - kama[i-1])
    # KAMA slope for trend
    kama_slope = np.diff(kama, prepend=kama[0])
    kama_slope_aligned = align_htf_to_ltf(prices, df_1w, kama_slope)
    
    # ATR for volatility and trailing stop
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (1.8x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    highest_high_since_entry = 0.0
    lowest_low_since_entry = 0.0
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(weekly_high_aligned[i]) or
            np.isnan(weekly_low_aligned[i]) or
            np.isnan(kama_slope_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # Trend filter from weekly KAMA slope
        bullish_trend = kama_slope_aligned[i] > 0
        bearish_trend = kama_slope_aligned[i] < 0
        
        # Volume confirmation (1.8x average)
        volume_surge = volume[i] > 1.8 * vol_ma[i]
        
        if position == 0:
            # Long: price breaks above previous week's high in bullish trend with volume surge
            if close[i] > weekly_high_aligned[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
                highest_high_since_entry = high[i]
            # Short: price breaks below previous week's low in bearish trend with volume surge
            elif close[i] < weekly_low_aligned[i] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
                lowest_low_since_entry = low[i]
        else:
            if position == 1:
                # Update highest high since entry
                if high[i] > highest_high_since_entry:
                    highest_high_since_entry = high[i]
                
                # Trailing stop: exit if price drops 2.5*ATR from highest high
                if close[i] < highest_high_since_entry - 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    highest_high_since_entry = 0.0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Update lowest low since entry
                if low[i] < lowest_low_since_entry:
                    lowest_low_since_entry = low[i]
                
                # Trailing stop: exit if price rises 2.5*ATR from lowest low
                if close[i] > lowest_low_since_entry + 2.5 * atr[i]:
                    signals[i] = 0.0
                    position = 0
                    lowest_low_since_entry = 0.0
                else:
                    signals[i] = -0.25
    
    return signals