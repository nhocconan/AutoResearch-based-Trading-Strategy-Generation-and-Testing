#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d EMA34 trend filter and volume confirmation.
# Long when price breaks above upper Donchian channel with volume > 1.3x average AND price > 1d EMA34.
# Short when price breaks below lower Donchian channel with volume > 1.3x average AND price < 1d EMA34.
# Exit on opposite Donchian level or trend reversal.
# Uses discrete position sizing (0.25) to minimize fee churn. Target: 12-37 trades/year.
# Works in bull markets via breakout continuation and in bear markets via faded rallies at resistance.
# 12h timeframe reduces trade frequency vs lower TFs, improving fee drag profile.

name = "12h_Donchian20_1dTrend_Volume_v1"
timeframe = "12h"
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
    
    # Get 12h data for Donchian channel calculation
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    
    # Calculate Donchian(20) on 12h: upper = max(high,20), lower = min(low,20)
    # Use rolling window with min_periods to avoid look-ahead
    high_series = pd.Series(high_12h)
    low_series = pd.Series(low_12h)
    upper_12h = high_series.rolling(window=20, min_periods=20).max().values
    lower_12h = low_series.rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 12h timeframe (already aligned as we used 12h data)
    # But we need to align to original timeframe (12h -> 12h is identity)
    # Actually, since we're using 12h data for 12h timeframe, no alignment needed
    # However, we need to align the 12h indicators to the 12h prices array
    # Since df_12h is already aligned to prices index via get_htf_data, we can use directly
    upper_12h_aligned = upper_12h
    lower_12h_aligned = lower_12h
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Volume filter: current volume > 1.3x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.3 * vol_ma)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(upper_12h_aligned[i]) or np.isnan(lower_12h_aligned[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above upper Donchian with volume confirmation AND price > 1d EMA34
            if close[i] > upper_12h_aligned[i] and volume_filter[i] and close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below lower Donchian with volume confirmation AND price < 1d EMA34
            elif close[i] < lower_12h_aligned[i] and volume_filter[i] and close[i] < ema34_1d_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price breaks below lower Donchian OR trend reversal (price < 1d EMA34)
            if close[i] < lower_12h_aligned[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price breaks above upper Donchian OR trend reversal (price > 1d EMA34)
            if close[i] > upper_12h_aligned[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals