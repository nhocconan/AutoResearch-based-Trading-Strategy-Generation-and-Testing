#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Camarilla pivot breakout with daily volume confirmation and trend filter.
# Uses prior day's Camarilla levels (H3/L3) as support/resistance.
# Long when price breaks above H3 with volume > 1.5x 20-period average.
# Short when price breaks below L3 with volume > 1.5x 20-period average.
# Trend filter: only trade long when price > 200-period EMA, short when price < 200-period EMA.
# Designed for low trade frequency (20-50/year) to minimize fee drag.
# Works in bull markets (breakouts above H3 in uptrend) and bear markets (breakouts below L3 in downtrend).
name = "4h_Camarilla_H3L3_Volume_TrendFilter"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get daily data for Camarilla levels (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate prior day's Camarilla levels (H3, L3)
    # Formula: H3 = close + 1.1*(high - low)/2, L3 = close - 1.1*(high - low)/2
    high_d = df_1d['high'].values
    low_d = df_1d['low'].values
    close_d = df_1d['close'].values
    
    camarilla_range = high_d - low_d
    h3 = close_d + 1.1 * camarilla_range / 2
    l3 = close_d - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 4h timeframe (use previous day's levels)
    h3_aligned = align_htf_to_ltf(prices, df_1d, h3)
    l3_aligned = align_htf_to_ltf(prices, df_1d, l3)
    
    # Calculate 200-period EMA for trend filter
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Calculate 20-period average volume for confirmation
    vol_ma_20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Wait for EMA_200 calculation
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(h3_aligned[i]) or np.isnan(l3_aligned[i]) or
            np.isnan(ema_200[i]) or np.isnan(vol_ma_20[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above 1.5x average
        vol_confirm = volume[i] > 1.5 * vol_ma_20[i]
        
        if position == 0:
            # Long: price breaks above H3 AND volume confirmation AND price > EMA200 (uptrend)
            long_breakout = close[i] > h3_aligned[i]
            if vol_confirm and long_breakout and close[i] > ema_200[i]:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below L3 AND volume confirmation AND price < EMA200 (downtrend)
            elif vol_confirm and close[i] < l3_aligned[i] and close[i] < ema_200[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below L3 OR trend turns down (price < EMA200)
            exit_condition = close[i] < l3_aligned[i] or close[i] < ema_200[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above H3 OR trend turns up (price > EMA200)
            exit_condition = close[i] > h3_aligned[i] or close[i] > ema_200[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals