#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel with volume > 1.3x average AND price > 1d EMA34.
# Short when price breaks below lower Donchian channel with volume > 1.3x average AND price < 1d EMA34.
# Exit on opposite Donchian breakout or trend reversal (price crosses 1d EMA34).
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 20-50 trades/year.
# Works in bull markets via breakout continuation and in bear markets via faded rallies at resistance.

name = "4h_Donchian20_1dTrend_Volume_v2"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for HTF trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Calculate Donchian(20) channels on primary timeframe
    # Upper channel: highest high of last 20 bars
    # Lower channel: lowest low of last 20 bars
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    upper_channel = high_series.rolling(window=20, min_periods=20).max().values
    lower_channel = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        if np.isnan(ema34_1d_aligned[i]) or np.isnan(upper_channel[i]) or np.isnan(lower_channel[i]) or \
           np.isnan(vol_ma[i]):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper Donchian with volume confirmation AND price > 1d EMA34
            if close[i] > upper_channel[i] and volume_filter[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower Donchian with volume confirmation AND price < 1d EMA34
            elif close[i] < lower_channel[i] and volume_filter[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below lower Donchian OR trend reversal (price < 1d EMA34)
            if close[i] < lower_channel[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above upper Donchian OR trend reversal (price > 1d EMA34)
            if close[i] > upper_channel[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals