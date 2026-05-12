#!/usr/bin/env python3
# 6h_DonchianBreakout_1dPivotDirection_VolumeConfirmation
# Hypothesis: On 6h timeframe, enter long when price breaks above 20-period Donchian high,
# daily pivot is bullish (close > pivot), and volume > 20-period average.
# Enter short when price breaks below 20-period Donchian low,
# daily pivot is bearish (close < pivot), and volume > 20-period average.
# Exit when price crosses the 1d EMA34 (trend reversal).
# Uses Donchian breakouts for trend capture, daily pivot for directional bias,
# and volume confirmation to avoid false breakouts. Designed for low-frequency,
# high-conviction trades in both bull and bear markets.
# Targets 15-25 trades/year for minimal fee drag.

name = "6h_DonchianBreakout_1dPivotDirection_VolumeConfirmation"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 40:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Load daily data for pivot and EMA34
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    daily_high = df_1d['high'].values
    daily_low = df_1d['low'].values
    daily_close = df_1d['close'].values
    
    # Calculate daily pivot point
    daily_pivot = (daily_high + daily_low + daily_close) / 3.0
    
    # Calculate 1d EMA34 for trend exit
    ema34_1d = pd.Series(daily_close).ewm(span=34, adjust=False, min_periods=34).mean().values
    
    # Align daily levels to 6h timeframe
    pivot_aligned = align_htf_to_ltf(prices, df_1d, daily_pivot)
    ema34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema34_1d)
    
    # Donchian channel (20-period) on 6h data
    high_series = pd.Series(high)
    low_series = pd.Series(low)
    donchian_high = high_series.rolling(window=20, min_periods=20).max().values
    donchian_low = low_series.rolling(window=20, min_periods=20).min().values
    
    # Volume confirmation: 20-period moving average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 40  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(pivot_aligned[i]) or np.isnan(ema34_1d_aligned[i]) or 
            np.isnan(donchian_high[i]) or np.isnan(donchian_low[i]) or 
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        pivot_val = pivot_aligned[i]
        ema1d_trend = ema34_1d_aligned[i]
        dc_high = donchian_high[i]
        dc_low = donchian_low[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above Donchian high, daily close > pivot (bullish), volume > 20MA
            if close[i] > dc_high and close[i] > pivot_val and volume[i] > vol_ma_val:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below Donchian low, daily close < pivot (bearish), volume > 20MA
            elif close[i] < dc_low and close[i] < pivot_val and volume[i] > vol_ma_val:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price closes below 1d EMA34 (trend reversal)
            if close[i] < ema1d_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price closes above 1d EMA34 (trend reversal)
            if close[i] > ema1d_trend:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals