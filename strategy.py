#!/usr/bin/env python3
"""
Hypothesis: 12h Camarilla R1/S1 breakout with 1d EMA50 trend filter and volume confirmation.
Long when price breaks above Camarilla R1 AND volume > 1.8x 20-period average AND price > 1d EMA50.
Short when price breaks below Camarilla S1 AND volume > 1.8x 20-period average AND price < 1d EMA50.
Exit when price crosses the 1d EMA50 in opposite direction.
Uses tight entry conditions (12-37 trades/year target) to minimize fee drag while capturing institutional pivot breaks in both bull and bear markets.
12h timeframe reduces noise, Camarilla levels provide institutional reference points.
"""

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
    
    # Get 1d data for Camarilla pivot calculation (HTF)
    df_1d = get_htf_data(prices, '1d')
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate 1d EMA50 for trend filter
    close_1d_series = pd.Series(close_1d)
    ema_50_1d = close_1d_series.ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Calculate Camarilla levels from previous 1d bar
    # R1 = close + 1.1*(high-low)/12, S1 = close - 1.1*(high-low)/12
    cam_r1 = close_1d + 1.1 * (high_1d - low_1d) / 12
    cam_s1 = close_1d - 1.1 * (high_1d - low_1d) / 12
    
    # Get 12h data for volume average (primary timeframe)
    df_12h = get_htf_data(prices, '12h')
    volume_12h = df_12h['volume'].values
    
    # Calculate volume average (20-period) on 12h
    volume_12h_series = pd.Series(volume_12h)
    volume_ma_12h = volume_12h_series.rolling(window=20, min_periods=20).mean().values
    
    # Align all indicators to 12h timeframe
    ema_50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema_50_1d)
    cam_r1_aligned = align_htf_to_ltf(prices, df_1d, cam_r1)
    cam_s1_aligned = align_htf_to_ltf(prices, df_1d, cam_s1)
    volume_ma_aligned = align_htf_to_ltf(prices, df_12h, volume_ma_12h)
    
    signals = np.zeros(n)
    position = 0  # -1: short, 0: flat, 1: long
    
    start_idx = 100  # warmup for indicators
    
    for i in range(start_idx, n):
        # Skip if any required data is not available
        if (np.isnan(ema_50_1d_aligned[i]) or np.isnan(cam_r1_aligned[i]) or 
            np.isnan(cam_s1_aligned[i]) or np.isnan(volume_ma_aligned[i])):
            signals[i] = 0.0
            continue
        
        ema_50 = ema_50_1d_aligned[i]
        r1 = cam_r1_aligned[i]
        s1 = cam_s1_aligned[i]
        vol_ma = volume_ma_aligned[i]
        vol = volume[i]
        price = close[i]
        high_price = high[i]
        low_price = low[i]
        
        if position == 0:
            # Long: price breaks above R1 AND volume > 1.8x avg AND price > 1d EMA50 (bullish trend)
            if high_price > r1 and vol > 1.8 * vol_ma and price > ema_50:
                signals[i] = 0.25
                position = 1
            # Short: price breaks below S1 AND volume > 1.8x avg AND price < 1d EMA50 (bearish trend)
            elif low_price < s1 and vol > 1.8 * vol_ma and price < ema_50:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: price crosses below 1d EMA50
            if price < ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: price crosses above 1d EMA50
            if price > ema_50:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals

name = "12h_Camarilla_R1S1_Volume_1dEMA50_Filter"
timeframe = "12h"
leverage = 1.0