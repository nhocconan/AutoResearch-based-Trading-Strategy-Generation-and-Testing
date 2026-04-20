#!/usr/bin/env python3
"""
4h_Keltner_Channel_Breakout_Volume_1d_Trend_Filter
Hypothesis: Trade breakouts above/below Keltner Channel (2x ATR) on 4h with volume confirmation and 1d EMA50 trend filter.
Keltner Channels adapt to volatility, providing dynamic support/resistance. In strong trends, price tends to stay
outside the channel, signaling continuation. Volume confirms breakout strength, and 1d trend filter avoids counter-trend trades.
Works in bull/bear: 1d trend filter ensures we only trade with the higher-timeframe trend, reducing whipsaw.
Target: 50-120 total trades over 4 years (12-30/year) with position size 0.25 to limit trade frequency and fee drag.
"""

name = "4h_Keltner_Channel_Breakout_Volume_1d_Trend_Filter"
timeframe = "4h"
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
    
    # Get 1d data ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
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
    
    # Calculate ATR(20) for Keltner Channel
    def atr(high, low, close, period):
        tr = np.zeros_like(high)
        tr[0] = high[0] - low[0]
        for i in range(1, len(high)):
            tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        
        atr_vals = np.zeros_like(tr)
        atr_vals[0] = tr[0]
        for i in range(1, len(tr)):
            atr_vals[i] = (atr_vals[i-1] * (period-1) + tr[i]) / period
        return atr_vals
    
    atr20 = atr(high, low, close, 20)
    
    # Calculate EMA20 for Keltner Channel middle line
    ema20 = ema(close, 20)
    
    # Keltner Channel: Upper = EMA20 + 2*ATR20, Lower = EMA20 - 2*ATR20
    kc_upper = ema20 + 2.0 * atr20
    kc_lower = ema20 - 2.0 * atr20
    
    # Volume spike: volume > 1.5x 20-period average
    vol_ma20 = np.full_like(volume, np.nan)
    for i in range(20, len(volume)):
        vol_ma20[i] = np.mean(volume[i-20:i])
    volume_spike = volume > (1.5 * vol_ma20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are ready
    
    for i in range(start_idx, n):
        # Skip if any required data is NaN
        if (np.isnan(ema50_1d_aligned[i]) or np.isnan(kc_upper[i]) or np.isnan(kc_lower[i]) or 
            np.isnan(ema20[i]) or np.isnan(close[i]) or np.isnan(volume[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above Keltner Upper with volume spike AND 1d uptrend (close > EMA50)
            if close[i] > kc_upper[i] and volume_spike[i] and close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Keltner Lower with volume spike AND 1d downtrend (close < EMA50)
            elif close[i] < kc_lower[i] and volume_spike[i] and close[i] < ema50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price crosses below EMA20 (middle line) OR 1d trend turns down
            if close[i] < ema20[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price crosses above EMA20 (middle line) OR 1d trend turns up
            if close[i] > ema20[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals