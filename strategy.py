#/usr/bin/env python3
"""
12h_WMA_Trend_Filter_With_Volume_and_RSI
Hypothesis: Trade 12h price direction using Weighted Moving Average (WMA) trend filter with volume confirmation and RSI filter.
Long when price > WMA(34) with volume > 1.5x average and RSI < 60; short when price < WMA(34) with volume > 1.5x average and RSI > 40.
Exit when price crosses back over WMA(34) or volume drops below average.
Uses 1w trend filter to avoid counter-trend trades in strong trends.
Designed for 12h timeframe to capture medium-term moves while reducing noise.
Target: 50-150 total trades over 4 years (12-37/year) with position size 0.25.
Works in bull/bear: 1w trend filter avoids counter-trend trades, volume filter reduces false signals.
"""

name = "12h_WMA_Trend_Filter_With_Volume_and_RSI"
timeframe = "12h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Get 12h data ONCE before loop
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 34:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    
    # Calculate WMA(34) on 12h close
    def wma(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            weights = np.arange(1, period + 1)
            weights_sum = weights.sum()
            for i in range(period-1, len(values)):
                result[i] = np.dot(values[i-period+1:i+1], weights) / weights_sum
        return result
    
    wma34_12h = wma(close_12h, 34)
    wma34_12h_aligned = align_htf_to_ltf(prices, df_12h, wma34_12h)
    
    # Get 1w data for trend filter
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 20:
        return np.zeros(n)
    
    close_1w = df_1w['close'].values
    
    # Calculate EMA(20) on 1w close for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema20_1w = ema(close_1w, 20)
    ema20_1w_aligned = align_htf_to_ltf(prices, df_1w, ema20_1w)
    
    # Calculate volume filter (volume > 1.5x 20-period average)
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Calculate RSI(14) on 12h close
    def rsi(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            delta = np.diff(values)
            up = np.where(delta > 0, delta, 0)
            down = np.where(delta < 0, -delta, 0)
            roll_up = np.full_like(values, np.nan)
            roll_down = np.full_like(values, np.nan)
            for i in range(period, len(values)):
                roll_up[i] = np.mean(up[i-period+1:i+1])
                roll_down[i] = np.mean(down[i-period+1:i+1])
            rs = np.where(roll_down != 0, roll_up / roll_down, 0)
            result[period-1:] = 100 - (100 / (1 + rs[period-1:]))
        return result
    
    rsi14_12h = rsi(close_12h, 14)
    rsi14_12h_aligned = align_htf_to_ltf(prices, df_12h, rsi14_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready (34 for WMA + buffer)
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(wma34_12h_aligned[i]) or np.isnan(ema20_1w_aligned[i]) or 
            np.isnan(rsi14_12h_aligned[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price > WMA34 with volume filter AND RSI < 60 AND 1w uptrend (close > EMA20)
            if close[i] > wma34_12h_aligned[i] and volume_filter[i] and rsi14_12h_aligned[i] < 60 and close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price < WMA34 with volume filter AND RSI > 40 AND 1w downtrend (close < EMA20)
            elif close[i] < wma34_12h_aligned[i] and volume_filter[i] and rsi14_12h_aligned[i] > 40 and close[i] < ema20_1w_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price < WMA34 OR volume drops below average OR 1w trend turns down
            if close[i] < wma34_12h_aligned[i] or not volume_filter[i] or close[i] < ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price > WMA34 OR volume drops below average OR 1w trend turns up
            if close[i] > wma34_12h_aligned[i] or not volume_filter[i] or close[i] > ema20_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals