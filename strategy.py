#!/usr/bin/env python3
# 1d_1w_1WeekLowBreakout_TrendFilter_Volume
# Hypothesis: Daily breakout above the previous week's high or below the previous week's low with weekly trend filter and volume confirmation. Works in bull via weekly high breakout in uptrend, and in bear via weekly low breakdown in downtrend. Designed for low trade frequency (7-25/year) to minimize fee drag.

name = "1d_1w_1WeekLowBreakout_TrendFilter_Volume"
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
    
    # Weekly trend: slope of 20-period EMA on weekly close
    weekly_close = df_1w['close'].values
    ema_20_1w = pd.Series(weekly_close).ewm(span=20, adjust=False, min_periods=20).mean().values
    ema_slope_20_1w = np.diff(ema_20_1w, prepend=ema_20_1w[0])
    ema_slope_20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema_slope_20_1w)
    
    # ATR for volatility and trailing stop
    tr1 = np.maximum(high - low, np.absolute(high - np.roll(close, 1)))
    tr2 = np.absolute(low - np.roll(close, 1))
    tr = np.maximum(tr1, tr2)
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(window=14, min_periods=14).mean().values
    
    # Volume confirmation (2.0x 20-period average)
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
            np.isnan(ema_slope_20_1w_aligned[i]) or
            np.isnan(atr[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
                highest_high_since_entry = 0.0
                lowest_low_since_entry = 0.0
            continue
        
        # Trend filter from weekly EMA20 slope
        bullish_trend = ema_slope_20_1w_aligned[i] > 0
        bearish_trend = ema_slope_20_1w_aligned[i] < 0
        
        # Volume confirmation (2.0x average)
        volume_surge = volume[i] > 2.0 * vol_ma[i]
        
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