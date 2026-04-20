#!/usr/bin/env python3
"""
12h_Volume_Weighted_Price_Action_With_1D_Trend_Filter
Hypothesis: Trade 12h price action using volume-weighted average price (VWAP) deviation with 1d trend filter.
Long when price closes above VWAP with volume expansion and 1d uptrend; short when price closes below VWAP with volume expansion and 1d downtrend.
Uses 12h VWAP calculated from typical price and volume, and 1d EMA50 for trend filtering.
Designed for 12h timeframe to capture medium-term swings with low trade frequency.
Target: 15-30 total trades over 4 years (4-8/year) with position size 0.25.
Works in bull/bear: 1d trend filter avoids counter-trend trades, volume filter reduces false signals.
"""

name = "12h_Volume_Weighted_Price_Action_With_1D_Trend_Filter"
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
    
    # Calculate 12h VWAP: cumulative (typical price * volume) / cumulative volume
    typical_price = (high + low + close) / 3.0
    price_volume = typical_price * volume
    cum_pv = np.cumsum(price_volume)
    cum_vol = np.cumsum(volume)
    vwap = np.divide(cum_pv, cum_vol, out=np.full_like(cum_pv, np.nan), where=cum_vol!=0)
    
    # Volume filter: volume > 1.5x 20-period average
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_filter = volume > (1.5 * vol_ma20)
    
    # Get 1d data for trend filter (using daily EMA50)
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    def ema(values, period):
        result = np.full_like(values, np.nan)
        if len(values) >= period:
            multiplier = 2.0 / (period + 1)
            result[period-1] = np.mean(values[:period])
            for i in range(period, len(values)):
                result[i] = multiplier * values[i] + (1 - multiplier) * result[i-1]
        return result
    
    ema50_1d = ema(close_1d, 50)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(vwap[i]) or np.isnan(ema50_1d_aligned[i]) or 
            np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price closes above VWAP with volume expansion AND 1d uptrend (close > EMA50)
            if close[i] > vwap[i] and volume_filter[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price closes below VWAP with volume expansion AND 1d downtrend (close < EMA50)
            elif close[i] < vwap[i] and volume_filter[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price closes below VWAP OR 1d trend turns down
            if close[i] < vwap[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price closes above VWAP OR 1d trend turns up
            if close[i] > vwap[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals