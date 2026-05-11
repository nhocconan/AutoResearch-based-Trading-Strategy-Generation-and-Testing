#!/usr/bin/env python3
name = "1h_Camarilla_R1S1_Breakout_4hTrend_1dVolume"
timeframe = "1h"
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
    
    # Get 4h data for Camarilla pivot calculation
    df_4h = get_htf_data(prices, '4h')
    if len(df_4h) < 1:
        return np.zeros(n)
    
    high_4h = df_4h['high'].values
    low_4h = df_4h['low'].values
    close_4h = df_4h['close'].values
    
    # Calculate previous day's Camarilla levels (using prior 4h bar's data)
    # For each 4h bar, we use the previous completed 4h bar's OHLC
    # We'll shift the arrays by 1 to get prior bar data
    if len(high_4h) < 2:
        return np.zeros(n)
    
    phigh_4h = np.roll(high_4h, 1)  # previous 4h bar high
    plow_4h = np.roll(low_4h, 1)    # previous 4h bar low
    pclose_4h = np.roll(close_4h, 1) # previous 4h bar close
    phigh_4h[0] = np.nan  # first bar has no previous
    plow_4h[0] = np.nan
    pclose_4h[0] = np.nan
    
    # Camarilla R1, S1 levels from previous 4h bar
    R1 = pclose_4h + (phigh_4h - plow_4h) * 1.1 / 12
    S1 = pclose_4h - (phigh_4h - plow_4h) * 1.1 / 12
    
    # Align Camarilla levels to 1h timeframe
    R1_aligned = align_htf_to_ltf(prices, df_4h, R1)
    S1_aligned = align_htf_to_ltf(prices, df_4h, S1)
    
    # Get 4h trend: EMA50 on 4h close
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_4h = close_4h > ema50_4h
    trend_up_4h_aligned = align_htf_to_ltf(prices, df_4h, trend_up_4h)
    
    # Get 1d volume filter: current volume > 1.5x 20-period average
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    vol_1d = df_1d['volume'].values
    vol_ma20_1d = pd.Series(vol_1d).rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = vol_1d > 1.5 * vol_ma20_1d
    volume_filter_1d_aligned = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    # Session filter: 8-20 UTC
    hours = pd.DatetimeIndex(prices['open_time']).hour
    session_filter = (hours >= 8) & (hours <= 20)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA50
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(R1_aligned[i]) or np.isnan(S1_aligned[i]) or
            np.isnan(trend_up_4h_aligned[i]) or np.isnan(volume_filter_1d_aligned[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: price breaks above R1 + 4h uptrend + volume confirmation + session
            if (close[i] > R1_aligned[i] and 
                trend_up_4h_aligned[i] and 
                volume_filter_1d_aligned[i] and 
                session_filter[i]):
                signals[i] = 0.20
                position = 1
            # Short: price breaks below S1 + 4h downtrend + volume confirmation + session
            elif (close[i] < S1_aligned[i] and 
                  not trend_up_4h_aligned[i] and 
                  volume_filter_1d_aligned[i] and 
                  session_filter[i]):
                signals[i] = -0.20
                position = -1
        elif position == 1:
            # Long exit: price breaks below S1 OR 4h trend turns down
            if (close[i] < S1_aligned[i] or 
                not trend_up_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # Short exit: price breaks above R1 OR 4h trend turns up
            if (close[i] > R1_aligned[i] or 
                trend_up_4h_aligned[i]):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals