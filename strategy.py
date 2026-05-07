#!/usr/bin/env python3
# 12h_Donchian20_Breakout_1dTrend_Volume
# Hypothesis: On 12h chart, enter long when price breaks above 20-period Donchian upper band with volume confirmation and daily trend up (close > EMA50), short when price breaks below lower band with volume confirmation and daily trend down (close < EMA50). Exit when price crosses the opposite band. Uses volume filter to avoid false breakouts and EMA50 for trend filter. Designed for low trade frequency (~15-25/year) to minimize fee drag and work in trending markets. Works in both bull and bear markets by capturing breakouts with volume and trend filters.

timeframe = "12h"
name = "12h_Donchian20_Breakout_1dTrend_Volume"
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
    
    # Donchian Channel parameters
    dc_period = 20
    
    # Calculate Donchian Bands
    dc_upper = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    dc_lower = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Daily trend filter: EMA50 on 1d
    df_1d = get_htf_data(prices, '1d')
    ema_50_1d = pd.Series(df_1d['close'].values).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(dc_period, n):
        # Skip if any critical value is NaN
        if (np.isnan(dc_upper[i]) or np.isnan(dc_lower[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema_50_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian upper band + volume spike + daily uptrend
            if close[i] > dc_upper[i] and volume[i] > 1.5 * vol_ma[i] and close[i] > ema_50_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian lower band + volume spike + daily downtrend
            elif close[i] < dc_lower[i] and volume[i] > 1.5 * vol_ma[i] and close[i] < ema_50_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price crosses below Donchian lower band (stoploss)
            if close[i] < dc_lower[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price crosses above Donchian upper band (stoploss)
            if close[i] > dc_upper[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals