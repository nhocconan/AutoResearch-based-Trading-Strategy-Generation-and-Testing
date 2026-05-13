#!/usr/bin/env python3
# Hypothesis: 1d Donchian(20) breakout with 1w trend filter (price > 1w EMA50) and volume confirmation.
# Long when price breaks above Donchian(20) high AND price > 1w EMA50 AND volume > 1.5x 20-day average volume.
# Short when price breaks below Donchian(20) low AND price < 1w EMA50 AND volume > 1.5x 20-day average volume.
# Exit when price touches the opposite Donchian(20) level (mean reversion) OR trend reversal (price crosses 1w EMA50).
# Uses 1d timeframe for lower frequency, Donchian for structure, 1w EMA for trend filter, volume for confirmation.
# Target: 30-100 total trades over 4 years (7-25/year). Works in bull via breakout continuation, bear via faded rallies.

name = "1d_Donchian20_1wTrend_Volume_v1"
timeframe = "1d"
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
    
    # Get 1d data for Donchian calculation (already 1d, but using helper for consistency)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    volume_1d = df_1d['volume'].values
    
    # Calculate Donchian(20) on 1d
    donchian_high = pd.Series(high_1d).rolling(window=20, min_periods=20).max().values
    donchian_low = pd.Series(low_1d).rolling(window=20, min_periods=20).min().values
    
    # Volume filter: current 1d volume > 1.5x 20-day average
    vol_ma_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = volume_1d > (1.5 * vol_ma_1d)
    
    # Get 1w data for EMA50 trend filter
    df_1w = get_htf_data(prices, '1w')
    close_1w = df_1w['close'].values
    
    # Calculate EMA(50) on 1w close for trend filter
    ema50_1w = pd.Series(close_1w).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1w_aligned = align_htf_to_ltf(prices, df_1w, ema50_1w)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):  # Start after sufficient data for all indicators
        # Skip if any required data is NaN
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or
            np.isnan(ema50_1w_aligned[i]) or np.isnan(vol_ma_1d[i])):
            signals[i] = 0.0
            continue
        
        if position == 0:
            # LONG: price breaks above Donchian(20) high AND price > 1w EMA50 AND volume confirmation
            if close[i] > donchian_high[i] and close[i] > ema50_1w_aligned[i] and volume_filter_1d[i]:
                signals[i] = 0.25
                position = 1
            # SHORT: price breaks below Donchian(20) low AND price < 1w EMA50 AND volume confirmation
            elif close[i] < donchian_low[i] and close[i] < ema50_1w_aligned[i] and volume_filter_1d[i]:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: price touches Donchian(20) low (mean reversion) OR trend reversal (price < 1w EMA50)
            if close[i] < donchian_low[i] or close[i] < ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: price touches Donchian(20) high (mean reversion) OR trend reversal (price > 1w EMA50)
            if close[i] > donchian_high[i] or close[i] > ema50_1w_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals