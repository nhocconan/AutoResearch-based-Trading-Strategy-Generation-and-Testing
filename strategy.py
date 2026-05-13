#!/usr/bin/env python3
name = "1H_Camarilla_R1S1_Breakout_4HTrend_1DVolume"
timeframe = "1h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    # Load 4H and 1D data ONCE before loop
    df_4h = get_htf_data(prices, '4h')
    df_1d = get_htf_data(prices, '1d')
    if len(df_4h) < 20 or len(df_1d) < 20:
        return np.zeros(n)
    
    # 4H EMA50 for trend filter
    close_4h = df_4h['close'].values
    ema50_4h = pd.Series(close_4h).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_4h_aligned = align_htf_to_ltf(prices, df_4h, ema50_4h)
    
    # 1D Volume average for volume filter
    volume_1d = df_1d['volume'].values
    vol_avg_1d = pd.Series(volume_1d).rolling(window=20, min_periods=20).mean().values
    vol_avg_1d_aligned = align_htf_to_ltf(prices, df_1d, vol_avg_1d)
    
    # Pre-calculate session hours (08-20 UTC) for filtering
    hours = pd.DatetimeIndex(prices['open_time']).hour
    
    # Calculate Camarilla levels from previous day's OHLC
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla R1, S1 levels (using previous day's data)
    camarilla_r1 = close_1d + (high_1d - low_1d) * 1.0833 / 12
    camarilla_s1 = close_1d - (high_1d - low_1d) * 1.0833 / 12
    
    # Align Camarilla levels to 1H timeframe
    camarilla_r1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_r1)
    camarilla_s1_aligned = align_htf_to_ltf(prices, df_1d, camarilla_s1)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(50, n):  # Start after sufficient data for indicators
        # Skip if any required data is NaN
        if (np.isnan(ema50_4h_aligned[i]) or np.isnan(camarilla_r1_aligned[i]) or 
            np.isnan(camarilla_s1_aligned[i]) or np.isnan(vol_avg_1d_aligned[i])):
            signals[i] = 0.0
            continue
        
        # Session filter: only trade 08-20 UTC
        hour = hours[i]
        if hour < 8 or hour > 20:
            signals[i] = 0.0 if position == 0 else (0.20 if position == 1 else -0.20)
            continue
        
        # Trend filter: price above/below 4H EMA50
        price_above_ema = prices['close'].iloc[i] > ema50_4h_aligned[i]
        price_below_ema = prices['close'].iloc[i] < ema50_4h_aligned[i]
        
        # Volume filter: current volume > 1.5x 1D average volume
        volume_filter = prices['volume'].iloc[i] > 1.5 * vol_avg_1d_aligned[i]
        
        if position == 0:
            # LONG: Price breaks above Camarilla R1 + uptrend + volume
            if (prices['close'].iloc[i] > camarilla_r1_aligned[i] and 
                price_above_ema and volume_filter):
                signals[i] = 0.20
                position = 1
            # SHORT: Price breaks below Camarilla S1 + downtrend + volume
            elif (prices['close'].iloc[i] < camarilla_s1_aligned[i] and 
                  price_below_ema and volume_filter):
                signals[i] = -0.20
                position = -1
            else:
                signals[i] = 0.0
        elif position == 1:
            # EXIT LONG: Price breaks below Camarilla S1 or trend weakens
            if (prices['close'].iloc[i] < camarilla_s1_aligned[i] or 
                not price_above_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.20
        elif position == -1:
            # EXIT SHORT: Price breaks above Camarilla R1 or trend weakens
            if (prices['close'].iloc[i] > camarilla_r1_aligned[i] or 
                not price_below_ema):
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.20
    
    return signals