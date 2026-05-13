#!/usr/bin/env python3
# Hypothesis: 4h Donchian(20) breakout with 12h EMA50 trend filter and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 12h EMA50 AND volume > 1.5x average.
# Short when price breaks below Donchian(20) low AND price < 12h EMA50 AND volume > 1.5x average.
# Exit when price crosses the opposite Donchian(20) level (long exit at Donchian low, short exit at Donchian high).
# Uses 4h primary timeframe for balance of signal frequency and fee drag, 12h EMA for trend filter.
# Target: 75-200 total trades over 4 years (19-50/year). Works in bull via breakout continuation, bear via faded rallies.

name = "4h_Donchian20_12hEMA50_Volume_v1"
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
    
    # Get 4h data for Donchian channels
    df_4h = get_htf_data(prices, '4h')
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    
    # Calculate Donchian(20) on 4h: highest high and lowest low over 20 periods
    # Using pandas rolling for efficiency and correct min_periods
    high_roll_4h = pd.Series(high_4h).rolling(window=20, min_periods=20).max().values
    low_roll_4h = pd.Series(low_4h).rolling(window=20, min_periods=20).min().values
    
    # Align Donchian levels to 4h timeframe (no additional delay needed for Donchian)
    donchian_high_4h_aligned = align_htf_to_ltf(prices, df_4h, high_roll_4h)
    donchian_low_4h_aligned = align_htf_to_ltf(prices, df_4h, low_roll_4h)
    
    # Get 12h data for EMA50 trend filter
    df_12h = get_htf_data(prices, '12h')
    close_12h = df_12h['close'].values
    volume_12h = df_12h['volume'].values
    
    # Calculate EMA(50) on 12h close for trend filter
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_12h_aligned = align_htf_to_ltf(prices, df_12h, ema50_12h)
    
    # Volume filter: current 12h volume > 1.5x 20-period average
    vol_ma_12h = pd.Series(volume_12h).rolling(window=20, min_periods=20).mean().values
    volume_filter_12h = volume_12h > (1.5 * vol_ma_12h)
    volume_filter_12h_aligned = align_htf_to_ltf(prices, df_12h, volume_filter_12h)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_high_4h_aligned[i]) or np.isnan(donchian_low_4h_aligned[i]) or
            np.isnan(ema50_12h_aligned[i]) or np.isnan(volume_filter_12h_aligned[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian high AND price > 12h EMA50 AND volume confirmation
            if close[i] > donchian_high_4h_aligned[i] and close[i] > ema50_12h_aligned[i] and volume_filter_12h_aligned[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian low AND price < 12h EMA50 AND volume confirmation
            elif close[i] < donchian_low_4h_aligned[i] and close[i] < ema50_12h_aligned[i] and volume_filter_12h_aligned[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price crosses below Donchian low (opposite breakout level)
            if close[i] < donchian_low_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price crosses above Donchian high (opposite breakout level)
            if close[i] > donchian_high_4h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals