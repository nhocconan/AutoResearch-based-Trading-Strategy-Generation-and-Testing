#!/usr/bin/env python3
"""
Hypothesis: 4h Camarilla R3/S3 breakout with 1d EMA34 trend filter and volume confirmation
- Uses 1d EMA34 to define higher timeframe trend: only trade breakouts in trend direction
- Camarilla R3/S3 levels on 4h provide precise entry/exit points with proven edge
- Volume confirmation (> 1.5x 20-period average) filters false breakouts
- Designed for 4h timeframe targeting 19-50 trades/year (75-200 over 4 years)
- Works in both bull and bear markets by trading with the daily trend
"""

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1d EMA34 for trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    ema_34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_34_1d)
    
    # Calculate 4h Camarilla levels (R3, S3)
    lookback = 20
    highest_high = pd.Series(high).rolling(window=lookback, min_periods=lookback).max().values
    lowest_low = pd.Series(low).rolling(window=lookback, min_periods=lookback).min().values
    close_prev = pd.Series(close).shift(1).values  # previous close for Camarilla calculation
    
    # Camarilla levels: R3 = close + 1.1*(high-low)/2, S3 = close - 1.1*(high-low)/2
    # Using lookback period high/low and previous close
    rang = highest_high - lowest_low
    r3 = close_prev + 1.1 * rang / 2
    s3 = close_prev - 1.1 * rang / 2
    
    # Volume confirmation: > 1.5x 20-period average
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Start from index where all indicators are ready
    start_idx = max(34, lookback, 20) + 1  # for EMA34, Camarilla, volume MA, and shift
    
    for i in range(start_idx, n):
        # Skip if data not ready
        if (np.isnan(ema_34_1d_aligned[i]) or 
            np.isnan(r3[i]) or np.isnan(s3[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above R3 with daily uptrend and volume
            long_breakout = (close[i] > r3[i] and 
                           close[i] > ema_34_1d_aligned[i] and
                           volume[i] > 1.5 * vol_ma[i])
            
            # Short conditions: price breaks below S3 with daily downtrend and volume
            short_breakout = (close[i] < s3[i] and 
                            close[i] < ema_34_1d_aligned[i] and
                            volume[i] > 1.5 * vol_ma[i])
            
            if long_breakout:
                signals[i] = 0.25
                position = 1
            elif short_breakout:
                signals[i] = -0.25
                position = -1
        else:
            # Exit conditions: opposite Camarilla breakout or trend reversal
            exit_signal = False
            
            if position == 1:
                # Exit long: price breaks below S3 or daily trend turns bearish
                if (close[i] < s3[i] or 
                    close[i] < ema_34_1d_aligned[i]):
                    exit_signal = True
            elif position == -1:
                # Exit short: price breaks above R3 or daily trend turns bullish
                if (close[i] > r3[i] or 
                    close[i] > ema_34_1d_aligned[i]):
                    exit_signal = True
            
            if exit_signal:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25 if position == 1 else -0.25
    
    return signals

name = "4h_Camarilla_R3S3_Breakout_1dEMA34_Trend_VolumeConfirm"
timeframe = "4h"
leverage = 1.0