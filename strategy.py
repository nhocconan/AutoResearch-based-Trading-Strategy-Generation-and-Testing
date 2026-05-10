#!/usr/bin/env python3
# 4h_1d_Camarilla_R1_S1_Breakout_1dTrend_Volume
# Hypothesis: Uses daily Camarilla R1/S1 levels for breakout direction, filtered by 1-day EMA34 trend and volume spike.
# Enters long when price breaks above R1 in bullish trend (close > EMA34) with volume > 1.5x 20-period average.
# Enters short when price breaks below S1 in bearish trend (close < EMA34) with volume confirmation.
# Exits when price closes back inside the previous day's range (between prior day's low and high).
# Designed for low trade frequency (<30/year) to minimize fee drag and work in both bull and bear markets.

name = "4h_1d_Camarilla_R1_S1_Breakout_1dTrend_Volume"
timeframe = "4h"
leverage = 1.0

import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

def generate_signals(prices):
    n = len(prices)
    if n < 50:
        return np.zeros(n)
    
    # Get 1d data for Camarilla levels and trend filter
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 35:
        return np.zeros(n)
    
    # 4h OHLCV
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Calculate 1-day EMA34 for trend filter
    close_1d = df_1d['close'].values
    ema_34 = pd.Series(close_1d).ewm(span=34, adjust=False, min_periods=34).mean().values
    ema_34_aligned = align_htf_to_ltf(prices, df_1d, ema_34)
    
    # Calculate daily Camarilla levels (R1, S1)
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d_vals = df_1d['close'].values
    
    # Camarilla R1 = close + 1.1*(high-low)/12
    # Camarilla S1 = close - 1.1*(high-low)/12
    r1 = close_1d_vals + 1.1 * (high_1d - low_1d) / 12
    s1 = close_1d_vals - 1.1 * (high_1d - low_1d) / 12
    
    # Align R1 and S1 to 4h timeframe
    r1_aligned = align_htf_to_ltf(prices, df_1d, r1)
    s1_aligned = align_htf_to_ltf(prices, df_1d, s1)
    
    # Previous day's high and low for exit condition (price inside prior day's range)
    prev_high = np.roll(high_1d, 1)
    prev_low = np.roll(low_1d, 1)
    # Set first value to NaN to avoid using uninitialized roll
    prev_high[0] = np.nan
    prev_low[0] = np.nan
    prev_high_aligned = align_htf_to_ltf(prices, df_1d, prev_high)
    prev_low_aligned = align_htf_to_ltf(prices, df_1d, prev_low)
    
    # Volume confirmation (1.5x 20-period average)
    vol_ma = pd.Series(volume).rolling(window=20, min_periods=20).mean().values
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    # Warmup
    start_idx = 50
    
    for i in range(start_idx, n):
        # Skip if any critical values are NaN
        if (np.isnan(ema_34_aligned[i]) or
            np.isnan(r1_aligned[i]) or
            np.isnan(s1_aligned[i]) or
            np.isnan(prev_high_aligned[i]) or
            np.isnan(prev_low_aligned[i]) or
            np.isnan(vol_ma[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        # Trend filter from 1-day EMA34
        bullish_trend = close[i] > ema_34_aligned[i]
        bearish_trend = close[i] < ema_34_aligned[i]
        
        # Volume confirmation (1.5x average)
        volume_surge = volume[i] > 1.5 * vol_ma[i]
        
        if position == 0:
            # Long: breakout above R1 in bullish trend with volume
            if close[i] > r1_aligned[i] and bullish_trend and volume_surge:
                signals[i] = 0.25
                position = 1
            # Short: breakdown below S1 in bearish trend with volume
            elif close[i] < s1_aligned[i] and bearish_trend and volume_surge:
                signals[i] = -0.25
                position = -1
        else:
            if position == 1:
                # Long exit: price closes back inside previous day's range
                if close[i] <= prev_high_aligned[i] and close[i] >= prev_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = 0.25
            elif position == -1:
                # Short exit: price closes back inside previous day's range
                if close[i] <= prev_high_aligned[i] and close[i] >= prev_low_aligned[i]:
                    signals[i] = 0.0
                    position = 0
                else:
                    signals[i] = -0.25
    
    return signals