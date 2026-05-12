#!/usr/bin/env python3
# 4h_InsideBar_Breakout_1dTrend_VolumeFilter
# Hypothesis: On 4h timeframe, enter long when price breaks above the high of an inside bar (narrow range) with price > daily EMA50 and volume > 1.5x 20-period MA.
# Enter short when price breaks below the low of an inside bar with price < daily EMA50 and volume > 1.5x MA.
# Exit when price crosses back to the opposite side of the inside bar's range.
# Uses daily trend filter and volume confirmation to avoid false breakouts. Targets 20-40 trades/year for low fee drag.
# Works in bull markets via breakouts and in bear markets via faded breaks of low-volatility consolidations.

name = "4h_InsideBar_Breakout_1dTrend_VolumeFilter"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Inside bar detection: current bar range within previous bar
    prev_high = np.roll(high, 1)
    prev_low = np.roll(low, 1)
    inside_bar = (high <= prev_high) & (low >= prev_low)
    # Mark the inside bar's high and low
    inside_high = np.where(inside_bar, high, np.nan)
    inside_low = np.where(inside_bar, low, np.nan)
    # Forward fill to get the most recent inside bar's levels
    inside_high_series = pd.Series(inside_high).ffill().values
    inside_low_series = pd.Series(inside_low).ffill().values
    
    # Daily EMA50 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    daily_close = df_1d['close'].values
    daily_ema50 = pd.Series(daily_close).ewm(span=50, adjust=False, min_periods=50).mean().values
    daily_ema50_aligned = align_htf_to_ltf(prices, df_1d, daily_ema50)
    
    # Volume confirmation: 20-period MA on 4h data
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Ensure indicators are stable
    
    for i in range(start_idx, n):
        # Skip if any critical data is not ready
        if (np.isnan(inside_high_series[i]) or np.isnan(inside_low_series[i]) or 
            np.isnan(daily_ema50_aligned[i]) or np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        ib_high = inside_high_series[i]
        ib_low = inside_low_series[i]
        daily_trend = daily_ema50_aligned[i]
        vol_ma_val = vol_ma[i]
        
        if position == 0:
            # LONG: Price breaks above inside bar high with trend and volume filter
            if close[i] > ib_high and close[i] > daily_trend and volume[i] > vol_ma_val * 1.5:
                signals[i] = 0.25
                position = 1
            # SHORT: Price breaks below inside bar low with trend and volume filter
            elif close[i] < ib_low and close[i] < daily_trend and volume[i] > vol_ma_val * 1.5:
                signals[i] = -0.25
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price crosses back below inside bar low (failed breakout)
            if close[i] < ib_low:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # EXIT SHORT: Price crosses back above inside bar high (failed breakout)
            if close[i] > ib_high:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals