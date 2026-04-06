#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray with Daily Trend Filter.
# Uses daily EMA50 as trend filter (bull if close > EMA50, bear if close < EMA50).
# Elder Ray: Bull Power = High - EMA13, Bear Power = Low - EMA13.
# Long: Bull Power > 0 and Bear Power < 0 in bull trend.
# Short: Bull Power < 0 and Bear Power > 0 in bear trend.
# Volume filter: current volume > 1.3x 20-period average.
# Designed to work in bull markets (via trend + bull power) and bear markets (via trend + bear power).
# Target: 50-150 trades over 4 years (12-37/year).

name = "6h_elder_ray_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 20:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA50 on daily closes
    ema50_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 50:
        ema50_1d[49] = np.mean(close_1d[:50])
        for i in range(50, len(close_1d)):
            ema50_1d[i] = (close_1d[i] * 2/51) + (ema50_1d[i-1] * 49/51)
    
    # Align EMA50 to 6h timeframe (shifted by 1 daily bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # EMA13 for Elder Ray (calculated on 6h closes)
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13[12] = np.mean(close[:13])
        for i in range(13, n):
            ema13[i] = (close[i] * 2/14) + (ema13[i-1] * 12/14)
    
    # Bull Power and Bear Power
    bull_power = high - ema13
    bear_power = low - ema13
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if EMA data not available
        if np.isnan(ema50_1d_aligned[i]) or np.isnan(ema13[i]) or np.isnan(vol_ma[i]):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Determine trend from daily EMA50
        is_uptrend = close[i] > ema50_1d_aligned[i]
        is_downtrend = close[i] < ema50_1d_aligned[i]
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.3
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: trend reversal or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.0 * atr_approx
            
            if (not is_uptrend or close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend reversal or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.0 * atr_approx
            
            if (not is_downtrend or close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume confirmation
            if volume_filter:
                # Long: bull power positive and bear power negative in uptrend
                if (bull_power[i] > 0 and bear_power[i] < 0 and is_uptrend):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: bull power negative and bear power positive in downtrend
                elif (bull_power[i] < 0 and bear_power[i] > 0 and is_downtrend):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals