#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

# Hypothesis: 1d strategy using weekly high/low breakout with volume confirmation and EMA200 trend filter.
# Buys when price breaks above weekly high with volume spike and above EMA200.
# Sells when price breaks below weekly low with volume spike and below EMA200.
# Exits when price crosses back below/above the weekly midpoint.
# Designed for low trade frequency (10-25/year) to minimize fee drag in bear markets.
# Weekly levels provide strong support/resistance that work in both trending and ranging markets.

name = "1d_1wHighLow_Breakout_VolumeTrend"
timeframe = "1d"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 200:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get weekly data
    df_1w = get_htf_data(prices, '1w')
    if len(df_1w) < 10:
        return np.zeros(n)
    
    high_1w = df_1w['high'].values
    low_1w = df_1w['low'].values
    close_1w = df_1w['close'].values
    
    # Calculate weekly high, low, and midpoint
    weekly_high = high_1w
    weekly_low = low_1w
    weekly_mid = (high_1w + low_1w) / 2.0
    
    # Calculate EMA200 on daily close for trend filter
    close_series = pd.Series(close)
    ema_200 = close_series.ewm(span=200, adjust=False, min_periods=200).mean().values
    
    # Align weekly indicators to daily timeframe
    weekly_high_aligned = align_htf_to_ltf(prices, df_1w, weekly_high)
    weekly_low_aligned = align_htf_to_ltf(prices, df_1w, weekly_low)
    weekly_mid_aligned = align_htf_to_ltf(prices, df_1w, weekly_mid)
    
    # Volume confirmation: daily volume spike (2x 20-period EMA)
    vol_ema = pd.Series(volume).ewm(span=20, adjust=False, min_periods=20).mean().values
    vol_spike = volume > (vol_ema * 2.0)
    
    signals = np.zeros(n)
    position = 0  # 0: flat, 1: long, -1: short
    
    start_idx = 200  # Ensure enough data for EMA200
    
    for i in range(start_idx, n):
        # Skip if any critical data is NaN
        if (np.isnan(weekly_high_aligned[i]) or 
            np.isnan(weekly_low_aligned[i]) or 
            np.isnan(weekly_mid_aligned[i]) or 
            np.isnan(ema_200[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        if position == 0:
            # Enter long: price breaks above weekly high + volume spike + above EMA200
            if close[i] > weekly_high_aligned[i] and vol_spike[i] and close[i] > ema_200[i]:
                signals[i] = 0.25
                position = 1
            # Enter short: price breaks below weekly low + volume spike + below EMA200
            elif close[i] < weekly_low_aligned[i] and vol_spike[i] and close[i] < ema_200[i]:
                signals[i] = -0.25
                position = -1
        elif position == 1:
            # Exit long: price crosses below weekly midpoint
            if close[i] < weekly_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        elif position == -1:
            # Exit short: price crosses above weekly midpoint
            if close[i] > weekly_mid_aligned[i]:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals