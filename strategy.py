#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 6-hour Elder Ray Bull/Bear Power with Trend Filter.
# Uses daily EMA200 as trend filter (price > EMA200 = bullish, < EMA200 = bearish).
# Elder Ray: Bull Power = High - EMA13, Bear Power = EMA13 - Low.
# Long when Bull Power > 0 and trend bullish, Short when Bear Power > 0 and trend bearish.
# Volume filter (current volume > 1.5x 20-period average) ensures quality signals.
# Works in bull/bear markets via trend alignment. Target: 75-200 trades over 4 years.

name = "6h_elder_ray_trend_filter_v1"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 30:
        return np.zeros(n)
    
    # Price and volume data
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Daily EMA200 for trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA200 on daily close
    ema200_1d = np.full(len(close_1d), np.nan)
    if len(close_1d) >= 200:
        ema200_1d[199] = np.mean(close_1d[:200])
        for i in range(200, len(close_1d)):
            ema200_1d[i] = (close_1d[i] * 2 / 201) + (ema200_1d[i-1] * (199 / 201))
    
    # Align EMA200 to 6h timeframe (shifted by 1 daily bar)
    ema200_aligned = align_htf_to_ltf(prices, df_1d, ema200_1d)
    
    # EMA13 for Elder Ray (on 6h close)
    ema13 = np.full(n, np.nan)
    if n >= 13:
        ema13[12] = np.mean(close[:13])
        for i in range(13, n):
            ema13[i] = (close[i] * 2 / 14) + (ema13[i-1] * (12 / 14))
    
    # Bull Power = High - EMA13, Bear Power = EMA13 - Low
    bull_power = high - ema13
    bear_power = ema13 - low
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = np.full(n, np.nan)
    for i in range(19, n):
        vol_ma[i] = np.mean(volume[i-19:i+1])
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    entry_price = 0.0
    
    for i in range(20, n):
        # Skip if EMA200 or EMA13 data not available
        if (np.isnan(ema200_aligned[i]) or np.isnan(ema13[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = position * 0.25
            else:
                signals[i] = 0.0
            continue
        
        # Volume condition
        volume_filter = volume[i] > vol_ma[i] * 1.5
        
        # Check exits and stoploss
        if position == 1:  # long position
            # Exit: trend turns bearish or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price - 2.5 * atr_approx
            
            if (close[i] < ema200_aligned[i] or 
                close[i] < stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:  # short position
            # Exit: trend turns bullish or stoploss
            atr_approx = max(high[i] - low[i], 0.001)
            stop_loss_level = entry_price + 2.5 * atr_approx
            
            if (close[i] > ema200_aligned[i] or 
                close[i] > stop_loss_level):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
        else:
            # Look for entries with volume and trend filter
            if volume_filter:
                # Long: Bull Power > 0 and trend bullish (price > EMA200)
                if (bull_power[i] > 0 and close[i] > ema200_aligned[i]):
                    signals[i] = 0.25
                    position = 1
                    entry_price = close[i]
                # Short: Bear Power > 0 and trend bearish (price < EMA200)
                elif (bear_power[i] > 0 and close[i] < ema200_aligned[i]):
                    signals[i] = -0.25
                    position = -1
                    entry_price = close[i]
    
    return signals