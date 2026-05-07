#!/usr/bin/env python3
# 12h_Donchian20_Breakout_1dTrend_Volume
# Hypothesis: On 12h chart, enter long when price breaks above 20-period Donchian high with 1d EMA34 uptrend and volume confirmation,
# enter short when price breaks below 20-period Donchian low with 1d EMA34 downtrend and volume confirmation.
# Use Donchian breakout as primary signal, 1d EMA34 for trend filter, volume spike for confirmation.
# Designed for low trade frequency (~15-30/year) to minimize fee drag and work in trending markets.
# Works in both bull and bear markets by capturing breakouts with volume and trend filters.
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
    
    # Calculate Donchian Channels
    dc_high = pd.Series(high).rolling(window=dc_period, min_periods=dc_period).max().values
    dc_low = pd.Series(low).rolling(window=dc_period, min_periods=dc_period).min().values
    
    # Volume spike: current volume > 1.5 * 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Get 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    ema_34_1d = pd.Series(df_1d['close'].values).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(dc_period, n):
        # Skip if any critical value is NaN
        if (np.isnan(dc_high[i]) or np.isnan(dc_low[i]) or 
            np.isnan(vol_ma[i]) or vol_ma[i] == 0 or
            np.isnan(ema_34_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long: price breaks above Donchian high + 1d EMA34 uptrend + volume spike
            if (close[i] > dc_high[i] and 
                ema_34_1d_aligned[i] > ema_34_1d_aligned[i-1] and
                volume[i] > 1.5 * vol_ma[i]):
                signals[i] = 0.25
                position = 1
            # Short: price breaks below Donchian low + 1d EMA34 downtrend + volume spike
            elif (close[i] < dc_low[i] and 
                  ema_34_1d_aligned[i] < ema_34_1d_aligned[i-1] and
                  volume[i] > 1.5 * vol_ma[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit: price breaks below Donchian low (stoploss)
            if close[i] < dc_low[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit: price breaks above Donchian high (stoploss)
            if close[i] > dc_high[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals