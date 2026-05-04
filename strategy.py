#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 12h Camarilla H3/L3 breakout with 1d EMA50 trend filter and volume spike
# Uses Camarilla pivot levels from daily chart (H3/L3) for tighter breakout levels than R3/S3.
# Enters long when price breaks above H3 with volume confirmation and 1d EMA50 uptrend.
# Enters short when price breaks below L3 with volume confirmation and 1d EMA50 downtrend.
# Exits when price re-enters the H3-L3 range or trend reverses.
# Designed for 12-37 trades/year (~50-150 total over 4 years) to minimize fee drag.
# H3/L3 levels provide stronger breakout signals than R3/S3, reducing false signals.
# Volume confirmation ensures breakout validity, EMA50 filters for trend alignment.
# Works in bull markets via breakouts and in bear markets via breakdowns.

name = "12h_Camarilla_H3L3_Breakout_1dEMA50_Trend_VolumeSpike"
timeframe = "12h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for Camarilla calculation - ONCE before loop
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 2:
        return np.zeros(n)
    
    high_1d = df_1d['high'].values
    low_1d = df_1d['low'].values
    close_1d = df_1d['close'].values
    
    # Calculate Camarilla levels for each 1d bar
    # H3 = close + 1.1*(high-low), L3 = close - 1.1*(high-low)
    camarilla_range = high_1d - low_1d
    h3_1d = close_1d + 1.1 * camarilla_range
    l3_1d = close_1d - 1.1 * camarilla_range
    
    # Align Camarilla levels to 12h timeframe (wait for completed 1d bar)
    h3_1d_aligned = align_htf_to_ltf(prices, df_1d, h3_1d)
    l3_1d_aligned = align_htf_to_ltf(prices, df_1d, l3_1d)
    
    # Get 1d data for EMA50 trend filter - ONCE before loop
    ema50_1d = pd.Series(close_1d).ewm(span=50, adjust=False, min_periods=50).mean().values
    
    # Align EMA50 to 12h timeframe (wait for completed 1d bar)
    ema50_1d_aligned = align_htf_to_ltf(prices, df_1d, ema50_1d)
    
    # Calculate volume spike filter (20-period volume MA)
    vol_ma_20 = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    volume_spike = volume > (vol_ma_20 * 2.0)  # Volume at least 2x average
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    for i in range(100, n):
        # Skip if any value is NaN
        if (np.isnan(h3_1d_aligned[i]) or np.isnan(l3_1d_aligned[i]) or 
            np.isnan(ema50_1d_aligned[i]) or np.isnan(volume_spike[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Long conditions: price breaks above H3 AND volume spike AND 1d EMA50 uptrend
            if (close[i] > h3_1d_aligned[i] and 
                volume_spike[i] and 
                close[i] > ema50_1d_aligned[i]):
                signals[i] = 0.25
                position = 1
            # Short conditions: price breaks below L3 AND volume spike AND 1d EMA50 downtrend
            elif (close[i] < l3_1d_aligned[i] and 
                  volume_spike[i] and 
                  close[i] < ema50_1d_aligned[i]):
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price re-enters Camarilla range (between L3 and H3) OR trend reverses
            if (close[i] >= l3_1d_aligned[i] and close[i] <= h3_1d_aligned[i]) or close[i] < ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price re-enters Camarilla range OR trend reverses
            if (close[i] >= l3_1d_aligned[i] and close[i] <= h3_1d_aligned[i]) or close[i] > ema50_1d_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals