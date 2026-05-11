#!/usr/bin/env python3
name = "4h_Camarilla_R3S3_Breakout_12hTrend_Volume"
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
    
    # Get 12h data for trend filter (EMA50)
    df_12h = get_htf_data(prices, '12h')
    if len(df_12h) < 50:
        return np.zeros(n)
    
    close_12h = df_12h['close'].values
    ema50_12h = pd.Series(close_12h).ewm(span=50, adjust=False, min_periods=50).mean().values
    trend_up_12h = close_12h > ema50_12h
    trend_up_12h_aligned = align_htf_to_ltf(prices, df_12h, trend_up_12h)
    
    # Get 1d data for Camarilla pivot levels
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 1:
        return np.zeros(n)
    
    # Calculate previous day's Camarilla levels
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Camarilla levels calculation
    range_1d = high_1d - low_1d
    camarilla_h5 = close_1d + range_1d * 1.1 / 2
    camarilla_h4 = close_1d + range_1d * 1.1 / 4
    camarilla_h3 = close_1d + range_1d * 1.1 / 6
    camarilla_l3 = close_1d - range_1d * 1.1 / 6
    camarilla_l4 = close_1d - range_1d * 1.1 / 4
    camarilla_l5 = close_1d - range_1d * 1.1 / 2
    
    # Align Camarilla levels to 4h timeframe
    camarilla_h3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h3)
    camarilla_l3_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l3)
    camarilla_h4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_h4)
    camarilla_l4_aligned = align_htf_to_ltf(prices, df_1d, camarilla_l4)
    
    # Volume confirmation: current volume > 1.8x 20-period average
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 1.8 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 20  # Need enough data for volume MA
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(trend_up_12h_aligned[i]) or 
            np.isnan(camarilla_h3_aligned[i]) or np.isnan(camarilla_l3_aligned[i]) or
            np.isnan(camarilla_h4_aligned[i]) or np.isnan(camarilla_l4_aligned[i]) or
            np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        if position == 0:
            # Long: Price breaks above H3 with 12h uptrend + volume confirmation
            if close[i] > camarilla_h3_aligned[i] and trend_up_12h_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Price breaks below L3 with 12h downtrend + volume confirmation
            elif close[i] < camarilla_l3_aligned[i] and not trend_up_12h_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Price breaks below L4 OR 12h trend turns down
            if close[i] < camarilla_l4_aligned[i] or not trend_up_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Price breaks above H4 OR 12h trend turns up
            if close[i] > camarilla_h4_aligned[i] or trend_up_12h_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals