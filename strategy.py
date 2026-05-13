#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 1d trend filter and volume confirmation.
# Long when price breaks above 20-period high AND close > 1d EMA50 AND volume > 1.5x 20-bar average
# Short when price breaks below 20-period low AND close < 1d EMA50 AND volume > 1.5x 20-bar average
# Exit when price crosses the 10-period mid-band OR trend reverses
# Uses 4h timeframe for optimal trade frequency, Donchian for structure, 1d EMA for trend filter, volume for confirmation.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull via breakout continuation, bear via faded rallies.

name = "4h_Donchian20_1dTrend_Volume_v1"
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
    
    # Calculate Donchian channels on primary timeframe (4h)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    mid_band = (highest_high + lowest_low) / 2.0
    
    # Volume filter: current volume > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > (1.5 * vol_ma)
    
    # Get 1d data for EMA50 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(50) on 1d close for trend filter
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(highest_high[i]) or np.isnan(lowest_low[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: Break above upper band AND price > 1d EMA50 AND volume confirmation
            if close[i] > highest_high[i] and close[i] > ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: Break below lower band AND price < 1d EMA50 AND volume confirmation
            elif close[i] < lowest_low[i] and close[i] < ema50_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses mid-band OR trend reversal (price < 1d EMA50)
            if close[i] < mid_band[i] or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses mid-band OR trend reversal (price > 1d EMA50)
            if close[i] > mid_band[i] or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals