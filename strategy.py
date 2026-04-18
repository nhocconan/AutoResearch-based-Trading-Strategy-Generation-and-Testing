#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 4h Donchian breakout with weekly trend filter and daily volume confirmation.
# Uses weekly EMA to determine trend direction (bull/bear) to avoid counter-trend trades.
# Only takes long trades in weekly uptrend (price > weekly EMA) and short trades in weekly downtrend (price < weekly EMA).
# Requires daily volume spike (volume > 1.5x daily average) for confirmation.
# Designed for low trade frequency (20-50/year) to minimize fee drag in 4h timeframe.
# Works in bull markets (breakouts above upper band in uptrend) and bear markets (breakouts below lower band in downtrend).
name = "4h_Donchian20_WeeklyTrend_DailyVolume"
timeframe = "4h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data for trend filter (ONCE before loop)
    df_1w = get_htf_data(prices, '1w')
    
    # Get daily data for volume filter (ONCE before loop)
    df_1d = get_htf_data(prices, '1d')
    
    # Calculate Donchian channels (20-period) using previous period's data to avoid look-ahead
    high_20 = pd.Series(high).rolling(window=20, min_periods=20).max().shift(1).values
    low_20 = pd.Series(low).rolling(window=20, min_periods=20).min().shift(1).values
    upper_band = high_20
    lower_band = low_20
    
    # Calculate weekly EMA (34-period) for trend filter
    close_w = df_1w['close'].values
    ema_w = pd.Series(close_w).ewm(span=34, adjust=False).values
    ema_w_aligned = align_htf_to_ltf(prices, df_1w, ema_w)
    
    # Calculate daily average volume (20-period) for confirmation
    vol_d = df_1d['volume'].values
    vol_ma_d = pd.Series(vol_d).rolling(window=20, min_periods=20).mean().values
    vol_ma_d_aligned = align_htf_to_ltf(prices, df_1d, vol_ma_d)
    
    # Session filter: 08-20 UTC
    hour_index = pd.DatetimeIndex(prices['open_time']).hour
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 100  # Wait for indicator calculations
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(upper_band[i]) or np.isnan(lower_band[i]) or
            np.isnan(ema_w_aligned[i]) or np.isnan(vol_ma_d_aligned[i])):
            signals[i] = 0.0
            continue
        
        hour = hour_index[i]
        in_session = 8 <= hour <= 20
        
        if not in_session:
            signals[i] = 0.0
            continue
        
        # Volume confirmation: current volume above 1.5x daily average
        vol_confirm = volume[i] > 1.5 * vol_ma_d_aligned[i]
        
        if position == 0:
            # Long: price breaks above upper band AND weekly uptrend AND volume confirmation
            long_breakout = close[i] > upper_band[i]
            weekly_uptrend = close[i] > ema_w_aligned[i]
            if vol_confirm and weekly_uptrend and long_breakout:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below lower band AND weekly downtrend AND volume confirmation
            elif vol_confirm and (close[i] < ema_w_aligned[i]) and close[i] < lower_band[i]:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Long exit: price falls below lower band OR weekly trend turns down
            exit_condition = close[i] < lower_band[i] or close[i] < ema_w_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Short exit: price rises above upper band OR weekly trend turns up
            exit_condition = close[i] > upper_band[i] or close[i] > ema_w_aligned[i]
            if exit_condition:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals