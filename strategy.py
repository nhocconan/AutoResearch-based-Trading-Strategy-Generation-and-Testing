#!/usr/bin/env python3
# Hypothesis: 12h Donchian(20) breakout with 1d trend filter, volume confirmation, and ATR-based stoploss.
# Long when price breaks above 20-period Donchian high AND price > 1d EMA34 AND volume > 1.5x average.
# Short when price breaks below 20-period Donchian low AND price < 1d EMA34 AND volume > 1.5x average.
# Exit when price touches the opposite Donchian channel (20-period low for long, high for short) OR trend reversal.
# Uses 12h timeframe for lower frequency, Donchian channels for structure, 1d EMA for trend filter, volume for confirmation.
# Target: 50-150 total trades over 4 years (12-37/year). Works in bull via breakout continuation, bear via faded rallies.

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
    
    # Get 12h data for Donchian channels and volume
    df_12h = get_htf_data(prices, '12h')
    high_12h = df_12h['high'].values
    low_12h = df_12h['low'].values
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate Donchian channels (20-period) on 12h
    donchian_high = pd.Series(high_12h).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_12h).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current 12h volume > 1.5x 20-period average
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_filter_12h = volume_12h > (1.5 * vol_ma_12h)
    
    # Get 1d data for EMA34 trend filter
    df_1d = get_htf_data(prices, '1d')
    close_1d = df_1d['close'].values
    
    # Calculate EMA(34) on 1d close for trend filter
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema34_1d_aligned[i]) or np.isnan(vol_ma_12h[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price > Donchian high AND price > 1d EMA34 AND volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema34_1d_aligned[i] and volume_filter_12h[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price < Donchian low AND price < 1d EMA34 AND volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema34_1d_aligned[i] and volume_filter_12h[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price < Donchian low (opposite channel) OR trend reversal (price < 1d EMA34)
            if close[i] < donchian_low[i] or close[i] < ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price > Donchian high (opposite channel) OR trend reversal (price > 1d EMA34)
            if close[i] > donchian_high[i] or close[i] > ema34_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals