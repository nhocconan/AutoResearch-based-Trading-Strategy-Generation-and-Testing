#!/usr/bin/env python3
name = "6h_WilliamsAlligator_1dTrend_Volume"
timeframe = "6h"
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
    
    # Get 1d data for Alligator (Williams) and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 13:
        return np.zeros(n)
    
    close_1d = df_1d['close'].values
    # Williams Alligator: Jaw (13), Teeth (8), Lips (5) SMAs of median price
    median_price_1d = (df_1d['high'].values + df_1d['low'].values) / 2
    jaw = pd.Series(median_price_1d).rolling(window=13, min_periods=13).mean().values  # Blue line
    teeth = pd.Series(median_price_1d).rolling(window=8, min_periods=8).mean().values    # Red line
    lips = pd.Series(median_price_1d).rolling(window=5, min_periods=5).mean().values    # Green line
    
    # Align to 6t timeframe
    jaw_aligned = align_htf_to_ltf(prices, df_1d, jaw)
    teeth_aligned = align_htf_to_ltf(prices, df_1d, teeth)
    lips_aligned = align_htf_to_ltf(prices, df_1d, lips)
    
    # Trend filter: 50 EMA on 1d close
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    trend_up_1d = close_1d > ema50_1d
    trend_up_1d_aligned = align_htf_to_ltf(prices, df_1d, trend_up_1d)
    
    # Volume confirmation: current volume > 2.0x 20-period average on 6h
    vol_ma20 = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    volume_filter = volume > 2.0 * vol_ma20
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 50  # Need enough data for EMA and indicators
    
    for i in range(start_idx, n):
        # Skip if any data is NaN
        if (np.isnan(jaw_aligned[i]) or np.isnan(teeth_aligned[i]) or np.isnan(lips_aligned[i]) or
            np.isnan(ema50_1d_aligned[i]) or np.isnan(vol_ma20[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.0
            continue
        
        # Alligator signals: Lips above Teeth above Jaw = bullish alignment
        bullish_aligned = lips_aligned[i] > teeth_aligned[i] and teeth_aligned[i] > jaw_aligned[i]
        # Bearish alignment: Lips below Teeth below Jaw
        bearish_aligned = lips_aligned[i] < teeth_aligned[i] and teeth_aligned[i] < jaw_aligned[i]
        
        if position == 0:
            # Long: Bullish Alligator alignment + 1d uptrend + volume confirmation
            if bullish_aligned and trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = 0.25
                position = 1
            # Short: Bearish Alligator alignment + 1d downtrend + volume confirmation
            elif bearish_aligned and not trend_up_1d_aligned[i] and volume_filter[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Long exit: Bearish Alligator alignment OR 1d trend turns down
            if bearish_aligned or not trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Short exit: Bullish Alligator alignment OR 1d trend turns up
            if bullish_aligned or trend_up_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals