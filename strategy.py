#!/usr/bin/env python3
name = "4h_Camarilla_R3_S3_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

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
    
    # Get 1d data for trend filter (EMA34) and Camarilla levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 34:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    
    # Calculate EMA34 on daily close
    ema34_1d = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    trend_up_1d = close_1d > ema34_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Calculate Camarilla levels from previous day
    # Camarilla: 
    # H5 = (high_prev + low_prev + 2*close_prev) + 1.1*(high_prev - low_prev)/2
    # H4 = (high_prev + low_prev + 2*close_prev) + 1.1*(high_prev - low_prev)/4
    # H3 = (high_prev + low_prev + 2*close_prev) + 1.1*(high_prev - low_prev)/6
    # L3 = (high_prev + low_prev + 2*close_prev) - 1.1*(high_prev - low_prev)/6
    # L4 = (high_prev + low_prev + 2*close_prev) - 1.1*(high_prev - low_prev)/4
    # L5 = (high_prev + low_prev + 2*close_prev) - 1.1*(high_prev - low_prev)/2
    
    prev_close = np.roll(close_1d, 1)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    
    # Set first day to avoid rollover issues
    prev_close[0] = close_1d[0]
    prev_high[0] = high_1d[0]
    prev_low[0] = low_1d[0]
    
    camarilla_base = (prev_high + prev_low + 2 * prev_close)
    camarilla_range = prev_high - prev_low
    
    H5 = camarilla_base + 1.1 * camarilla_range / 2
    H4 = camarilla_base + 1.1 * camarilla_range / 4
    H3 = camarilla_base + 1.1 * camarilla_range / 6
    L3 = camarilla_base - 1.1 * camarilla_range / 6
    L4 = camarilla_base - 1.1 * camarilla_range / 4
    L5 = camarilla_base - 1.1 * camarilla_range / 2
    
    # Align Camarilla levels to 4h timeframe
    H5_aligned = align_htf_to_ltf(prices, df_1d, H5)
    H4_aligned = align_htf_to_ltf(prices, df_1d, H4)
    H3_aligned = align_htf_to_ltf(prices, df_1d, H3)
    L3_aligned = align_htf_to_ltf(prices, df_1d, L3)
    L4_aligned = align_htf_to_ltf(prices, df_1d, L4)
    L5_aligned = align_htf_to_ltf(prices, df_1d, L5)
    
    # Volume confirmation: current volume > 1.5x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.5 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(H3_aligned[i]) or np.isnan(L3_aligned[i]) or 
            np.isnan(trend_up_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above H3 with daily uptrend and volume confirmation
            if close[i] > H3_aligned[i] and trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L3 with daily downtrend and volume confirmation
            elif close[i] < L3_aligned[i] and not trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below L3 or daily trend turns down
            if close[i] < L3_aligned[i] or not trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above H3 or daily trend turns up
            if close[i] > H3_aligned[i] or trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals