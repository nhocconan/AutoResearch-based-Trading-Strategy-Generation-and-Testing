#!/usr/bin/env python3
# Hypothesis: 6h Chandelier Exit trend reversal with 1d ADX trend filter and volume confirmation
# Long when price closes above Chandelier Exit long stop with 1d ADX > 25 and volume > 1.5x average
# Short when price closes below Chandelier Exit short stop with 1d ADX > 25 and volume > 1.5x average
# Exit when price crosses the Chandelier Exit in the opposite direction
# Chandelier Exit adapts to volatility, reducing whipsaw in choppy markets. ADX filters for trending conditions only.
# Designed for medium-frequency trades on 6h timeframe suitable for trending markets with controlled drawdowns
# Target: 50-150 total trades over 4 years (12-37/year) with size 0.25

name = "6h_ChandelierExit_ADX_Trend_Volume"
timeframe = "6h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    high = prices['high'].values
    low = prices['low'].values
    close = prices['close'].values
    volume = prices['volume'].values
    
    # Calculate Chandelier Exit (22-period, ATR multiplier 3.0)
    atr_period = 22
    atr_mult = 3.0
    
    # True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    tr1[0] = 0
    tr2[0] = 0
    tr3[0] = 0
    tr = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # ATR using Wilder's smoothing (equivalent to RMA)
    atr = np.zeros_like(tr)
    atr[atr_period-1] = np.mean(tr[:atr_period])
    for i in range(atr_period, len(tr)):
        atr[i] = (atr[i-1] * (atr_period - 1) + tr[i]) / atr_period
    
    # Chandelier Exit components
    highest_high = np.maximum.accumulate(high)
    lowest_low = np.minimum.accumulate(low)
    
    long_stop = highest_high - atr_mult * atr
    short_stop = lowest_low + atr_mult * atr
    
    # Calculate 1d ADX for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 30:
        return np.zeros(n)
    
    # ADX calculation (14-period)
    adx_period = 14
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # True Range and Directional Movement
    tr_1d = np.maximum(high_1d - low_1d, 
                       np.maximum(np.abs(high_1d - np.roll(close_1d, 1)), 
                                  np.abs(low_1d - np.roll(close_1d, 1))))
    tr_1d[0] = 0
    
    plus_dm = np.where((high_1d - np.roll(high_1d, 1)) > (np.roll(low_1d, 1) - low_1d), 
                       np.maximum(high_1d - np.roll(high_1d, 1), 0), 0)
    minus_dm = np.where((np.roll(low_1d, 1) - low_1d) > (high_1d - np.roll(high_1d, 1)), 
                        np.maximum(np.roll(low_1d, 1) - low_1d, 0), 0)
    plus_dm[0] = 0
    minus_dm[0] = 0
    
    # Smoothed TR, +DM, -DM
    atr_1d = np.zeros_like(tr_1d)
    atr_1d[adx_period-1] = np.mean(tr_1d[:adx_period])
    plus_dm_smooth = np.zeros_like(tr_1d)
    minus_dm_smooth = np.zeros_like(tr_1d)
    
    for i in range(adx_period, len(tr_1d)):
        atr_1d[i] = (atr_1d[i-1] * (adx_period - 1) + tr_1d[i]) / adx_period
        plus_dm_smooth[i] = (plus_dm_smooth[i-1] * (adx_period - 1) + plus_dm[i]) / adx_period
        minus_dm_smooth[i] = (minus_dm_smooth[i-1] * (adx_period - 1) + minus_dm[i]) / adx_period
    
    # Avoid division by zero
    plus_di = np.where(atr_1d != 0, 100 * plus_dm_smooth / atr_1d, 0)
    minus_di = np.where(atr_1d != 0, 100 * minus_dm_smooth / atr_1d, 0)
    dx = np.where((plus_di + minus_di) != 0, 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di), 0)
    
    # ADX smoothing
    adx = np.zeros_like(dx)
    if len(dx) >= adx_period:
        adx[adx_period-1] = np.mean(dx[:adx_period])
        for i in range(adx_period, len(dx)):
            adx[i] = (adx[i-1] * (adx_period - 1) + dx[i]) / adx_period
    
    adx_1d_aligned = align_htf_to_ltf(prices, df_1d, adx)
    
    # Volume confirmation: current volume > 1.5x 30-period average
    vol_ma = pd.Series(volume).rolling(window=30, min_periods=30).mean()
    vol_confirm = volume > (1.5 * vol_ma.values)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = max(50, 30)  # Need enough data for indicators
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(long_stop[i]) or np.isnan(short_stop[i]) or 
            np.isnan(adx_1d_aligned[i]) or np.isnan(vol_confirm[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price closes above Chandelier Exit long stop, ADX > 25, volume spike
            if (close[i] > long_stop[i] and 
                adx_1d_aligned[i] > 25 and 
                vol_confirm[i]):
                signals[i] = 0.25
                position = 1
            # Enter short: price closes below Chandelier Exit short stop, ADX > 25, volume spike
            elif (close[i] < short_stop[i] and 
                  adx_1d_aligned[i] > 25 and 
                  vol_confirm[i]):
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price closes below Chandelier Exit long stop
            if close[i] < long_stop[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price closes above Chandelier Exit short stop
            if close[i] > short_stop[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals