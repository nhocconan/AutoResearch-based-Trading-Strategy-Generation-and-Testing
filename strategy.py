#!/usr/bin/env python3
# 4h_MultiTimeframe_Structure_Breakout
# Hypothesis: Combines 4h price action with 1d trend filter and volume confirmation.
# Long when: 4h close > 4h Donchian high(20) AND 1d close > 1d EMA50 AND volume > 1.5x 20-period average.
# Short when: 4h close < 4h Donchian low(20) AND 1d close < 1d EMA50 AND volume > 1.5x 20-period average.
# Exit when: 4h close crosses 4h EMA20 (trend change).
# Uses tight conditions to limit trades (~25-35/year) and works in both bull (breakouts) and bear (mean reversion via trend filter).

name = "4h_MultiTimeframe_Structure_Breakout"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 60:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # 4h Donchian channels (20-period)
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # 4h EMA20 for exit
    close_series = pd.Series(close)
    ema20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean().values
    
    # Load daily data for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 50:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    
    # 1-day EMA50 for trend filter
    ema50_1d = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 60  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(ema20[i]) or np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        donch_high = donchian_high[i]
        donch_low = donchian_low[i]
        ema20_val = ema20[i]
        ema1d_trend = ema50_1d_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: 4h breakout above Donchian high + 1d uptrend + volume surge
            if close[i] > donch_high and close[i] > ema1d_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: 4h breakdown below Donchian low + 1d downtrend + volume surge
            elif close[i] < donch_low and close[i] < ema1d_trend and volume[i] > 1.5 * vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: 4h close below EMA20 (trend change)
            if close[i] < ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: 4h close above EMA20 (trend change)
            if close[i] > ema20_val:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals