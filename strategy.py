#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel with volume > 1.3x average AND price > 12h EMA50.
# Short when price breaks below lower Donchian channel with volume > 1.3x average AND price < 12h EMA50.
# Exit on opposite Donchian level or trend reversal.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Works in bull markets via breakout continuation and in bear markets via faded rallies at resistance.
# 4h timeframe balances trade frequency and signal quality, proven in DB for Donchian-based strategies.

name = "4h_Donchian20_12hTrend_Volume_v1"
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
    
    # Get 4h data for Donchian channel calculation
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate Donchian(20) on 4h: upper = max(high,20), lower = min(low,20)
    high_series = pd.Series(high_4h)
    low_series = pd.Series(low_4h)
    upper_4h = high_series.rolling(window=20, min_periods=20).max().values
    lower_4h = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to original timeframe (4h -> 4h is identity)
    upper_4h_aligned = upper_4h
    lower_4h_aligned = lower_4h
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    
    # Calculate EMA(50) on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(upper_4h_aligned[i]) or np.isnan(lower_4h_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper Donchian with volume confirmation AND price > 12h EMA50
            if close[i] > upper_4h_aligned[i] and volume_filter[i] and close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower Donchian with volume confirmation AND price < 12h EMA50
            elif close[i] < lower_4h_aligned[i] and volume_filter[i] and close[i] < ema50_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below lower Donchian OR trend reversal (price < 12h EMA50)
            if close[i] < lower_4h_aligned[i] or close[i] < ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above upper Donchian OR trend reversal (price > 12h EMA50)
            if close[i] > upper_4h_aligned[i] or close[i] > ema50_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals