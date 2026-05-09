#!/usr/bin/env python3
import numpy as np
import pandas as pd
from mtf_data import get_htf_data, align_htf_to_ltf

name = "6h_ElderRay_BullBearPower_1dTrend_Volume"
timeframe = "6h"
leverage = 1.0

def generate_signals(prices):
    n = len(prices)
    if n < 100:
        return np.zeros(n)
    
    close = prices['close'].values
    high = prices['high'].values
    low = prices['low'].values
    volume = prices['volume'].values
    
    # Get 1d data for EMA13 trend and Elder Ray calculations
    df_1d = get_htf_data(prices, '1d')
    if len(df_1d) < 20:
        return np.zeros(n)
    
    # Calculate EMA13 for 1d close (trend filter)
    close_1d = df_1d['close'].values
    ema13_1d = pd.Series(close_1d).ewm(span=13, adjust=False, min_periods=13).mean().values
    
    # Calculate Elder Ray components for 1d
    # Bull Power = High - EMA13
    # Bear Power = Low - EMA13
    bull_power = df_1d['high'].values - ema13_1d
    bear_power = df_1d['low'].values - ema13_1d
    
    # Volume filter: current 1d volume > 1.3 * 20-day average
    vol_series = pd.Series(df_1d['volume'].values)
    vol_ma = vol_series.rolling(window=20, min_periods=20).mean().values
    volume_filter_1d = df_1d['volume'].values > (vol_ma * 1.3)
    
    # Align all to 6h
    ema13_1d_6h = align_htf_to_ltf(prices, df_1d, ema13_1d)
    bull_power_6h = align_htf_to_ltf(prices, df_1d, bull_power)
    bear_power_6h = align_htf_to_ltf(prices, df_1d, bear_power)
    volume_filter_6h = align_htf_to_ltf(prices, df_1d, volume_filter_1d)
    
    signals = np.zeros(n)
    position = 0
    
    start_idx = 20  # Need enough data for EMA13 and volume MA
    
    for i in range(start_idx, n):
        if (np.isnan(ema13_1d_6h[i]) or np.isnan(bull_power_6h[i]) or
            np.isnan(bear_power_6h[i]) or np.isnan(volume_filter_6h[i])):
            if position != 0:
                signals[i] = 0.0
                position = 0
            continue
        
        trend = ema13_1d_6h[i]
        bull = bull_power_6h[i]
        bear = bear_power_6h[i]
        vol_filter = volume_filter_6h[i]
        
        if position == 0:
            # Enter long: Bull Power > 0 (bullish momentum) with volume and above EMA13
            if bull > 0 and close[i] > trend and vol_filter:
                signals[i] = 0.25
                position = 1
            # Enter short: Bear Power < 0 (bearish momentum) with volume and below EMA13
            elif bear < 0 and close[i] < trend and vol_filter:
                signals[i] = -0.25
                position = -1
        
        elif position == 1:
            # Exit long: Bear Power turns negative (momentum shift)
            if bear < 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = 0.25
        
        elif position == -1:
            # Exit short: Bull Power turns positive (momentum shift)
            if bull > 0:
                signals[i] = 0.0
                position = 0
            else:
                signals[i] = -0.25
    
    return signals